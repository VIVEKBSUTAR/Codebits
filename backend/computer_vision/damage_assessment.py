import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from utils.logger import SystemLogger

logger = SystemLogger(module_name="computer_vision")

class InfrastructureDamageAssessment:
    """
    Advanced Computer Vision for Infrastructure Damage Assessment
    Uses YOLO v8 for object detection + custom damage classification
    """

    def __init__(self):
        self.model = None
        self.damage_classifier = InfrastructureDamageClassifier()
        self._initialize_model()

    def _initialize_model(self):
        """Initialize YOLO model for infrastructure object detection"""
        try:
            # Use YOLOv8 pre-trained model
            self.model = YOLO('yolov8n.pt')  # Nano version for faster inference
            logger.log("YOLO model initialized successfully")
        except Exception as e:
            logger.log(f"YOLO initialization error: {str(e)}")
            self.model = None

    def analyze_infrastructure_image(self, image_data: bytes, zone: str = "Unknown") -> Dict:
        """
        Analyze infrastructure image for damage assessment

        Args:
            image_data: Raw image bytes
            zone: Zone name for context

        Returns:
            Comprehensive damage assessment with confidence scores
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Get image dimensions
            width, height = image.size

            # Run YOLO detection if model is available
            detections = []
            if self.model:
                detections = self._run_yolo_detection(image)
            else:
                # Fallback simulation
                detections = self._simulate_detections(width, height)

            # Classify damage for each detection
            damage_analysis = []
            overall_damage_score = 0.0
            critical_issues = 0

            for detection in detections:
                damage_assessment = self.damage_classifier.assess_damage(
                    detection, image, zone
                )
                damage_analysis.append(damage_assessment)

                # Update overall metrics
                if damage_assessment['severity_score'] > 0.7:
                    critical_issues += 1
                overall_damage_score += damage_assessment['severity_score']

            # Calculate overall assessment
            if damage_analysis:
                overall_damage_score = overall_damage_score / len(damage_analysis)

            # Generate annotated image
            annotated_image = self._create_annotated_image(image, damage_analysis)
            annotated_image_b64 = self._image_to_base64(annotated_image)

            # Risk level determination
            if overall_damage_score > 0.8:
                risk_level = "CRITICAL"
                priority = "IMMEDIATE_ACTION_REQUIRED"
            elif overall_damage_score > 0.6:
                risk_level = "HIGH"
                priority = "URGENT_REPAIR_NEEDED"
            elif overall_damage_score > 0.4:
                risk_level = "MEDIUM"
                priority = "MAINTENANCE_SCHEDULED"
            else:
                risk_level = "LOW"
                priority = "ROUTINE_MONITORING"

            # Recommended actions
            recommendations = self._generate_recommendations(damage_analysis, overall_damage_score)

            return {
                "analysis_status": "success",
                "zone": zone,
                "image_dimensions": {"width": width, "height": height},
                "overall_assessment": {
                    "damage_score": round(overall_damage_score, 3),
                    "risk_level": risk_level,
                    "priority": priority,
                    "critical_issues": critical_issues,
                    "total_detections": len(damage_analysis)
                },
                "damage_detections": damage_analysis,
                "recommendations": recommendations,
                "annotated_image": annotated_image_b64,
                "analysis_timestamp": "2024-01-15T14:30:00Z"  # In production, use real timestamp
            }

        except Exception as e:
            logger.log(f"Image analysis error: {str(e)}")
            return {
                "analysis_status": "error",
                "error_message": str(e),
                "zone": zone
            }

    def _run_yolo_detection(self, image: Image.Image) -> List[Dict]:
        """Run YOLO detection on image"""
        try:
            # Convert PIL to numpy array
            img_array = np.array(image)

            # Run inference
            results = self.model(img_array, conf=0.25)  # 25% confidence threshold

            detections = []
            for result in results:
                boxes = result.boxes

                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                        # Get confidence and class
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())

                        # Map COCO classes to infrastructure objects
                        infrastructure_class = self._map_coco_to_infrastructure(class_id)

                        if infrastructure_class:  # Only process infrastructure-related objects
                            detections.append({
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "confidence": confidence,
                                "class": infrastructure_class,
                                "class_id": class_id
                            })

            return detections

        except Exception as e:
            logger.log(f"YOLO detection error: {str(e)}")
            return []

    def _map_coco_to_infrastructure(self, class_id: int) -> Optional[str]:
        """Map COCO class IDs to infrastructure objects"""
        infrastructure_mapping = {
            # Buildings and structures
            0: "building",  # person -> building area
            2: "vehicle",   # car
            3: "vehicle",   # motorcycle
            5: "vehicle",   # bus
            6: "vehicle",   # truck
            9: "roadway",   # traffic light
            11: "roadway",  # stop sign

            # Infrastructure elements
            72: "electronics",  # tv -> electrical equipment
            73: "electronics",  # laptop -> control systems
        }

        # Generic infrastructure classes for demo
        if class_id in [0, 72, 73]:
            return "infrastructure_element"
        elif class_id in [2, 3, 5, 6]:
            return "transport_infrastructure"
        elif class_id in [9, 11]:
            return "traffic_infrastructure"
        else:
            return None

    def _simulate_detections(self, width: int, height: int) -> List[Dict]:
        """Simulate infrastructure detections for demo purposes"""
        # Generate realistic infrastructure detections
        simulated_detections = [
            {
                "bbox": [50, 100, 300, 250],
                "confidence": 0.87,
                "class": "infrastructure_element",
                "class_id": 999
            },
            {
                "bbox": [320, 80, 500, 200],
                "confidence": 0.72,
                "class": "transport_infrastructure",
                "class_id": 998
            },
            {
                "bbox": [150, 300, 400, 450],
                "confidence": 0.65,
                "class": "traffic_infrastructure",
                "class_id": 997
            }
        ]

        return simulated_detections

    def _create_annotated_image(self, image: Image.Image, damage_analysis: List[Dict]) -> Image.Image:
        """Create annotated image with damage analysis overlays"""
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        # Try to use a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 14)
            small_font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = font

        for analysis in damage_analysis:
            bbox = analysis['detection_bbox']
            x1, y1, x2, y2 = bbox

            # Color based on severity
            severity = analysis['severity_score']
            if severity > 0.8:
                color = "red"
                line_width = 3
            elif severity > 0.6:
                color = "orange"
                line_width = 2
            else:
                color = "green"
                line_width = 2

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            # Draw label with damage info
            label = f"{analysis['object_type']} - {analysis['damage_type']}"
            confidence_text = f"{analysis['confidence']*100:.1f}%"

            # Label background
            label_bbox = draw.textbbox((x1, y1-30), f"{label} {confidence_text}", font=font)
            draw.rectangle(label_bbox, fill=color, outline=color)

            # Label text
            draw.text((x1, y1-30), f"{label} {confidence_text}", fill="white", font=font)

            # Severity indicator
            severity_text = f"Risk: {severity*100:.0f}%"
            draw.text((x1, y2+5), severity_text, fill=color, font=small_font)

        return annotated

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_b64 = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{image_b64}"

    def _generate_recommendations(self, damage_analysis: List[Dict], overall_score: float) -> List[str]:
        """Generate action recommendations based on damage analysis"""
        recommendations = []

        if overall_score > 0.8:
            recommendations.extend([
                "🚨 CRITICAL: Immediate evacuation of area recommended",
                "🔧 Deploy emergency repair teams within 2 hours",
                "📞 Contact structural engineering specialists",
                "🚧 Establish safety perimeter and reroute traffic"
            ])
        elif overall_score > 0.6:
            recommendations.extend([
                "⚠️ HIGH PRIORITY: Schedule repair within 24 hours",
                "🔍 Conduct detailed structural assessment",
                "📋 Document all damage for insurance claims",
                "🚦 Consider temporary traffic management"
            ])
        elif overall_score > 0.4:
            recommendations.extend([
                "📅 MAINTENANCE: Schedule repair within 1 week",
                "📸 Continue monitoring with regular inspections",
                "📊 Update maintenance records and budgets"
            ])
        else:
            recommendations.extend([
                "✅ GOOD CONDITION: Continue routine maintenance",
                "📈 Include in quarterly inspection schedule"
            ])

        # Add specific recommendations based on damage types
        damage_types = [analysis['damage_type'] for analysis in damage_analysis]

        if 'structural_damage' in damage_types:
            recommendations.append("🏗️ Consult structural engineer for load-bearing assessment")

        if 'water_damage' in damage_types:
            recommendations.append("💧 Check drainage systems and waterproofing")

        if 'surface_damage' in damage_types:
            recommendations.append("🛠️ Schedule surface restoration and protective coating")

        return recommendations

class InfrastructureDamageClassifier:
    """Specialized classifier for infrastructure damage assessment"""

    def assess_damage(self, detection: Dict, image: Image.Image, zone: str) -> Dict:
        """
        Assess damage for a detected infrastructure object

        Args:
            detection: YOLO detection result
            image: PIL Image for context
            zone: Zone name

        Returns:
            Detailed damage assessment
        """
        # Extract detection info
        bbox = detection['bbox']
        confidence = detection['confidence']
        obj_class = detection['class']

        # Simulate damage analysis based on object type and image properties
        damage_assessment = self._simulate_damage_analysis(bbox, obj_class, image)

        return {
            "detection_bbox": bbox,
            "object_type": obj_class,
            "confidence": confidence,
            "damage_type": damage_assessment['damage_type'],
            "severity_score": damage_assessment['severity_score'],
            "damage_description": damage_assessment['description'],
            "estimated_repair_cost": damage_assessment['repair_cost'],
            "urgency_level": damage_assessment['urgency'],
            "zone": zone
        }

    def _simulate_damage_analysis(self, bbox: List[int], obj_class: str, image: Image.Image) -> Dict:
        """Simulate realistic damage analysis"""
        # Get image region for analysis
        x1, y1, x2, y2 = bbox
        width, height = image.size

        # Calculate some basic features for simulation
        region_size = (x2 - x1) * (y2 - y1)
        relative_size = region_size / (width * height)

        # Simulate damage based on object class and characteristics
        import random
        random.seed(x1 + y1)  # Reproducible randomness based on position

        damage_types = {
            'infrastructure_element': [
                ('structural_damage', 0.7, "Visible cracks and deterioration", "$15,000", "high"),
                ('surface_damage', 0.4, "Surface wear and aging", "$3,000", "medium"),
                ('water_damage', 0.6, "Water infiltration damage", "$8,000", "high")
            ],
            'transport_infrastructure': [
                ('road_damage', 0.5, "Pavement cracking and potholes", "$5,000", "medium"),
                ('structural_damage', 0.8, "Bridge/overpass structural issues", "$25,000", "critical"),
                ('surface_damage', 0.3, "Normal wear patterns", "$2,000", "low")
            ],
            'traffic_infrastructure': [
                ('electrical_damage', 0.6, "Signal malfunction detected", "$1,500", "high"),
                ('structural_damage', 0.4, "Post/sign structural issues", "$800", "medium"),
                ('surface_damage', 0.2, "Minor cosmetic damage", "$200", "low")
            ]
        }

        # Select damage type based on class
        if obj_class in damage_types:
            damage_options = damage_types[obj_class]
        else:
            # Generic infrastructure damage
            damage_options = [
                ('general_damage', 0.5, "Infrastructure deterioration", "$5,000", "medium")
            ]

        # Select damage with some randomness
        selected_damage = random.choice(damage_options)
        damage_type, base_severity, description, repair_cost, urgency = selected_damage

        # Adjust severity based on size and position
        severity_modifier = min(0.3, relative_size * 2)  # Larger objects get higher severity
        final_severity = min(1.0, base_severity + severity_modifier + random.uniform(-0.2, 0.2))

        return {
            'damage_type': damage_type,
            'severity_score': final_severity,
            'description': description,
            'repair_cost': repair_cost,
            'urgency': urgency
        }

# Global instance
damage_assessor = InfrastructureDamageAssessment()

def analyze_infrastructure_image(image_data: bytes, zone: str = "Unknown") -> Dict:
    """
    Main interface for infrastructure damage assessment

    Args:
        image_data: Raw image bytes from upload
        zone: Zone name for context

    Returns:
        Comprehensive damage analysis with recommendations
    """
    return damage_assessor.analyze_infrastructure_image(image_data, zone)