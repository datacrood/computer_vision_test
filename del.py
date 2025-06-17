import supervision as sv
import numpy as np
from ultralytics import YOLO
from collections import deque
from typing import Optional, List, Tuple, Dict, Any
import cv2

class EnhancedBallTracker:
    def __init__(
        self, 
        buffer_size: int = 10,
        speed_threshold: float = 500.0,  # pixels per frame
        confidence_threshold: float = 0.5,
        max_distance_threshold: float = 200.0,  # max pixels from centroid
        min_detections_for_speed: int = 2
    ):
        self.buffer = deque(maxlen=buffer_size)
        self.position_buffer = deque(maxlen=buffer_size)
        self.frame_numbers = deque(maxlen=buffer_size)
        self.confidence_buffer = deque(maxlen=buffer_size)
        self.speed_buffer = deque(maxlen=buffer_size - 1)
        
        # Thresholds
        self.speed_threshold = speed_threshold
        self.confidence_threshold = confidence_threshold
        self.max_distance_threshold = max_distance_threshold
        self.min_detections_for_speed = min_detections_for_speed
        
        # Frame tracking
        self.current_frame = 0
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'outliers_removed': 0,
            'speed_outliers': 0,
            'distance_outliers': 0,
            'confidence_outliers': 0
        }

    def calculate_speed(self, pos1: np.ndarray, pos2: np.ndarray, frame_diff: int = 1) -> float:
        """Calculate speed between two positions in pixels per frame"""
        if frame_diff <= 0:
            return 0.0
        distance = np.linalg.norm(pos2 - pos1)
        return distance / frame_diff

    def is_speed_outlier(self, current_pos: np.ndarray) -> bool:
        """Check if current position creates unrealistic speed"""
        if len(self.position_buffer) < self.min_detections_for_speed:
            return False
        
        prev_pos = self.position_buffer[-1]
        prev_frame = self.frame_numbers[-1]
        frame_diff = self.current_frame - prev_frame
        
        if frame_diff <= 0:
            return False
            
        speed = self.calculate_speed(prev_pos, current_pos, frame_diff)
        return speed > self.speed_threshold

    def is_distance_outlier(self, current_pos: np.ndarray) -> bool:
        """Check if current position is too far from trajectory centroid"""
        if len(self.position_buffer) < 2:
            return False
        
        # Calculate centroid of recent positions
        recent_positions = np.array(list(self.position_buffer))
        centroid = np.mean(recent_positions, axis=0)
        distance = np.linalg.norm(current_pos - centroid)
        
        return distance > self.max_distance_threshold

    def filter_detections(self, detections: sv.Detections) -> sv.Detections:
        """Filter detections based on confidence and other criteria"""
        if len(detections) == 0:
            return detections
        
        # Filter by confidence
        high_conf_mask = detections.confidence >= self.confidence_threshold
        
        if not np.any(high_conf_mask):
            # If no high confidence detections, keep the highest confidence one
            best_idx = np.argmax(detections.confidence)
            return detections[[best_idx]]
        
        return detections[high_conf_mask]

    def select_best_detection(self, detections: sv.Detections) -> Optional[sv.Detections]:
        """Select the best detection from multiple candidates"""
        if len(detections) == 0:
            return None
        
        if len(detections) == 1:
            return detections
        
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        
        # If we have trajectory history, use it to select best detection
        if len(self.position_buffer) > 0:
            centroid = np.mean(np.array(list(self.position_buffer)), axis=0)
            distances = np.linalg.norm(xy - centroid, axis=1)
            best_idx = np.argmin(distances)
            return detections[[best_idx]]
        
        # Otherwise, select highest confidence
        best_idx = np.argmax(detections.confidence)
        return detections[[best_idx]]

    def update_buffers(self, detection: sv.Detections):
        """Update all tracking buffers with new detection"""
        if len(detection) == 0:
            return
        
        center = detection.get_anchors_coordinates(sv.Position.CENTER)[0]
        confidence = detection.confidence[0]
        
        # Update buffers
        self.buffer.append(detection.get_anchors_coordinates(sv.Position.CENTER))
        self.position_buffer.append(center)
        self.frame_numbers.append(self.current_frame)
        self.confidence_buffer.append(confidence)
        
        # Calculate and store speed if possible
        if len(self.position_buffer) >= 2:
            prev_pos = list(self.position_buffer)[-2]
            prev_frame = list(self.frame_numbers)[-2]
            frame_diff = self.current_frame - prev_frame
            speed = self.calculate_speed(prev_pos, center, frame_diff)
            self.speed_buffer.append(speed)

    def update(self, detections: sv.Detections, frame_number: Optional[int] = None) -> sv.Detections:
        """Main update method with outlier removal"""
        if frame_number is not None:
            self.current_frame = frame_number
        else:
            self.current_frame += 1
        
        self.stats['total_detections'] += len(detections)
        
        if len(detections) == 0:
            return detections
        
        # Step 1: Filter by confidence
        filtered_detections = self.filter_detections(detections)
        
        # Step 2: Check for outliers
        valid_detections = []
        
        for i in range(len(filtered_detections)):
            single_detection = filtered_detections[[i]]
            center = single_detection.get_anchors_coordinates(sv.Position.CENTER)[0]
            confidence = single_detection.confidence[0]
            
            is_outlier = False
            
            # Check confidence outlier
            if confidence < self.confidence_threshold:
                is_outlier = True
                self.stats['confidence_outliers'] += 1
            
            # Check speed outlier
            elif self.is_speed_outlier(center):
                is_outlier = True
                self.stats['speed_outliers'] += 1
            
            # Check distance outlier
            elif self.is_distance_outlier(center):
                is_outlier = True
                self.stats['distance_outliers'] += 1
            
            if not is_outlier:
                valid_detections.append(i)
        
        # Step 3: Select best detection from valid ones
        if valid_detections:
            valid_detections_obj = filtered_detections[valid_detections]
            best_detection = self.select_best_detection(valid_detections_obj)
        else:
            # If all are outliers, skip this frame or use fallback logic
            self.stats['outliers_removed'] += len(filtered_detections)
            return sv.Detections.empty()
        
        # Step 4: Update buffers
        if best_detection and len(best_detection) > 0:
            self.update_buffers(best_detection)
            return best_detection
        
        return sv.Detections.empty()

    def get_trajectory(self) -> List[Tuple[float, float]]:
        """Get the current trajectory as list of (x, y) coordinates"""
        return [(pos[0], pos[1]) for pos in self.position_buffer]

    def get_speeds(self) -> List[float]:
        """Get the speed history"""
        return list(self.speed_buffer)

    def get_stats(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        stats = self.stats.copy()
        if stats['total_detections'] > 0:
            stats['outlier_percentage'] = (stats['outliers_removed'] / stats['total_detections']) * 100
        else:
            stats['outlier_percentage'] = 0
        return stats

    def reset(self):
        """Reset the tracker"""
        self.buffer.clear()
        self.position_buffer.clear()
        self.frame_numbers.clear()
        self.confidence_buffer.clear()
        self.speed_buffer.clear()
        self.current_frame = 0
        self.stats = {
            'total_detections': 0,
            'outliers_removed': 0,
            'speed_outliers': 0,
            'distance_outliers': 0,
            'confidence_outliers': 0
        }


class VersatileVisualizer:
    """Versatile visualization class for ball tracking results"""
    
    def __init__(
        self,
        show_trajectory: bool = True,
        show_speed: bool = True,
        show_confidence: bool = True,
        trajectory_length: int = 30,
        trajectory_thickness: int = 3
    ):
        self.show_trajectory = show_trajectory
        self.show_speed = show_speed
        self.show_confidence = show_confidence
        self.trajectory_length = trajectory_length
        self.trajectory_thickness = trajectory_thickness
        
        # Annotation components
        self.box_annotator = sv.BoxAnnotator(
            color=sv.Color.GREEN,
            thickness=2
        )
        self.label_annotator = sv.LabelAnnotator(
            color=sv.Color.GREEN,
            text_color=sv.Color.WHITE,
            text_scale=0.7,
            text_thickness=2
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=sv.Color.RED,
            thickness=self.trajectory_thickness,
            trace_length=self.trajectory_length
        )
        
        # Custom colors for different confidence levels
        self.confidence_colors = {
            'high': sv.Color.GREEN,    # >= 0.8
            'medium': sv.Color.YELLOW, # 0.5-0.8
            'low': sv.Color.RED        # < 0.5
        }

    def get_confidence_color(self, confidence: float) -> sv.Color:
        """Get color based on confidence level"""
        if confidence >= 0.8:
            return self.confidence_colors['high']
        elif confidence >= 0.5:
            return self.confidence_colors['medium']
        else:
            return self.confidence_colors['low']

    def create_labels(
        self, 
        detections: sv.Detections, 
        tracker: EnhancedBallTracker,
        frame_number: int
    ) -> List[str]:
        """Create informative labels for detections"""
        labels = []
        
        for i, (class_id, confidence) in enumerate(zip(detections.class_id, detections.confidence)):
            label_parts = []
            
            # Basic info
            label_parts.append(f"Ball: {confidence:.2f}")
            
            # Speed info
            if self.show_speed and len(tracker.get_speeds()) > 0:
                current_speed = tracker.get_speeds()[-1]
                label_parts.append(f"Speed: {current_speed:.1f}")
            
            # Frame number
            label_parts.append(f"Frame: {frame_number}")
            
            labels.append(" | ".join(label_parts))
        
        return labels

    def draw_trajectory_info(
        self, 
        frame: np.ndarray, 
        tracker: EnhancedBallTracker
    ) -> np.ndarray:
        """Draw additional trajectory information on frame"""
        if not self.show_trajectory:
            return frame
        
        trajectory = tracker.get_trajectory()
        if len(trajectory) < 2:
            return frame
        
        # Draw trajectory line
        points = np.array(trajectory, dtype=np.int32)
        
        for i in range(1, len(points)):
            # Fade color based on age
            alpha = i / len(points)
            color = (0, int(255 * alpha), 0)  # Green with fading
            cv2.line(frame, tuple(points[i-1]), tuple(points[i]), color, 2)
        
        # Draw speed graph (mini visualization)
        if self.show_speed and len(tracker.get_speeds()) > 1:
            self.draw_speed_graph(frame, tracker.get_speeds())
        
        return frame

    def draw_speed_graph(self, frame: np.ndarray, speeds: List[float]) -> None:
        """Draw a mini speed graph on the frame"""
        if len(speeds) < 2:
            return
        
        # Position for mini graph (top-right corner)
        graph_w, graph_h = 200, 100
        start_x = frame.shape[1] - graph_w - 20
        start_y = 20
        
        # Background
        cv2.rectangle(frame, (start_x, start_y), (start_x + graph_w, start_y + graph_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (start_x, start_y), (start_x + graph_w, start_y + graph_h), (255, 255, 255), 2)
        
        # Scale speeds to graph height
        max_speed = max(speeds) if speeds else 1
        scaled_speeds = [(speed / max_speed) * (graph_h - 20) for speed in speeds[-20:]]  # Last 20 speeds
        
        # Draw speed line
        points = []
        for i, speed in enumerate(scaled_speeds):
            x = start_x + 10 + int((i / len(scaled_speeds)) * (graph_w - 20))
            y = start_y + graph_h - 10 - int(speed)
            points.append((x, y))
        
        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], (0, 255, 0), 2)
        
        # Add labels
        cv2.putText(frame, f"Speed: {speeds[-1]:.1f}", (start_x + 5, start_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def annotate_frame(
        self, 
        frame: np.ndarray, 
        detections: sv.Detections, 
        tracker: EnhancedBallTracker,
        frame_number: int
    ) -> np.ndarray:
        """Main annotation method"""
        annotated_frame = frame.copy()
        
        if len(detections) == 0:
            return self.draw_trajectory_info(annotated_frame, tracker)
        
        # Create dynamic box annotator based on confidence
        if self.show_confidence and len(detections.confidence) > 0:
            confidence = detections.confidence[0]
            color = self.get_confidence_color(confidence)
            dynamic_box_annotator = sv.BoxAnnotator(color=color, thickness=2)
            dynamic_label_annotator = sv.LabelAnnotator(color=color, text_color=sv.Color.WHITE)
        else:
            dynamic_box_annotator = self.box_annotator
            dynamic_label_annotator = self.label_annotator
        
        # Annotate with boxes and labels
        annotated_frame = dynamic_box_annotator.annotate(annotated_frame, detections)
        
        labels = self.create_labels(detections, tracker, frame_number)
        annotated_frame = dynamic_label_annotator.annotate(annotated_frame, detections, labels=labels)
        
        # Add trajectory visualization
        annotated_frame = self.draw_trajectory_info(annotated_frame, tracker)
        
        return annotated_frame


# Usage Example
def process_video_with_enhanced_tracking(
    video_path: str,
    output_path: str,
    model_path: str,
    **tracker_kwargs
):
    """Process video with enhanced ball tracking"""
    
    # Initialize components
    model = YOLO(model_path)
    tracker = EnhancedBallTracker(**tracker_kwargs)
    visualizer = VersatileVisualizer()
    
    # Video info
    video_info = sv.VideoInfo.from_video_path(video_path)
    width, height = video_info.resolution_wh
    
    # Border filtering
    percentage_to_exclude_from_sides = 0.01
    border_x = width * percentage_to_exclude_from_sides
    border_y = height * percentage_to_exclude_from_sides
    
    all_detections = []
    frame_number = 0
    
    def callback(frame: np.ndarray, index) -> np.ndarray:
        nonlocal frame_number
        frame_number += 1
        
        # Run YOLO detection
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter detections (borders, class, etc.)
        detections = detections[
            (detections.xyxy[:, 0] > border_x) & 
            (detections.xyxy[:, 1] > border_y) & 
            (detections.xyxy[:, 2] < width - border_x) & 
            (detections.xyxy[:, 3] < height - border_y)
        ]
        
        # Keep only highest confidence detection initially
        if len(detections) > 1:
            sort_keep_indices = np.argsort(detections.confidence)[::-1][:1]
            detections = detections[sort_keep_indices]
        
        # Update tracker with outlier removal
        tracked_detections = tracker.update(detections, frame_number)
        
        # Visualize
        annotated_frame = visualizer.annotate_frame(
            frame, tracked_detections, tracker, frame_number
        )
        
        all_detections.append(tracked_detections)
        
        return annotated_frame
    
    # Process video
    sv.process_video(
        source_path=video_path,
        target_path=output_path,
        callback=callback
    )
    
    # Print statistics
    stats = tracker.get_stats()
    print("\n=== Tracking Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return all_detections, tracker

import supervision as sv
import numpy as np
from ultralytics import YOLO

# Your file paths
VIDEO_PATH = '/Users/spectatr/Downloads/nsl_match2/683efcc10388617b7ff4b9cf/input_001a21f5-580a-4280-98f2-49d4e3fbd28d1748958688.mp4'
OUTPUT_VIDEO_PATH = '/Users/spectatr/Downloads/testing2.mp4'
MODEL_PATH = '/Users/spectatr/Downloads/best_datav10_rectified_nano_1920_e261_final.pt'

# Initialize enhanced tracker with custom parameters
tracker = EnhancedBallTracker(
    buffer_size=10,
    speed_threshold=500.0,  # Adjust based on your sport (pixels per frame)
    confidence_threshold=0.3,  # Lower threshold for initial filtering
    max_distance_threshold=200.0,  # Max distance from trajectory centroid
    min_detections_for_speed=2
)

# Initialize versatile visualizer
visualizer = VersatileVisualizer(
    show_trajectory=True,
    show_speed=True,
    show_confidence=True,
    trajectory_length=30
)

# Alternative: Use the convenience function
all_detections, tracker = process_video_with_enhanced_tracking(
    video_path=VIDEO_PATH,
    output_path=OUTPUT_VIDEO_PATH,
    model_path=MODEL_PATH,
    buffer_size=10,
    speed_threshold=500.0,
    confidence_threshold=0.3
)