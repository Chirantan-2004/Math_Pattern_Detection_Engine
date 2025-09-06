"""
Kolam Mathematical Principle Identifier - Enhanced MVP 2
=====================================================
An interactive tool to analyze mathematical principles in Kolam patterns
including Fibonacci sequences, grid classifications, and complexity scoring.

Author: AI Assistant
Target: Educational and Research Use
Compatible: Google Colab
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Polygon
import math
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# ================================
# CORE DATA STRUCTURES
# ================================

class GridType(Enum):
    """Enumeration of different grid types found in Kolam patterns"""
    SQUARE = "Square Grid"
    TRIANGULAR = "Triangular Grid" 
    HEXAGONAL = "Hexagonal Grid"
    CIRCULAR = "Circular Grid"
    IRREGULAR = "Irregular Pattern"

@dataclass
class Point:
    """Represents a point in 2D space"""
    x: float
    y: float
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate Euclidean distance to another point"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class KolamPattern:
    """Represents a complete Kolam pattern with all its properties"""
    name: str
    dots: List[Point]
    connections: List[Tuple[int, int]]  # Indices of connected dots
    grid_type: GridType = GridType.IRREGULAR
    
@dataclass
class AnalysisResult:
    """Contains the complete analysis results of a Kolam pattern"""
    pattern_name: str
    fibonacci_score: float
    grid_type: GridType
    complexity_score: float
    symmetry_axes: int
    spiral_arms: int
    growth_ratio: float
    educational_explanation: str
    detailed_metrics: Dict[str, Any]

# ================================
# MATHEMATICAL ANALYSIS ENGINE
# ================================

class FibonacciAnalyzer:
    """Analyzes Fibonacci sequences and spiral patterns in Kolam designs"""
    
    @staticmethod
    def generate_fibonacci_sequence(n: int) -> List[int]:
        """Generate Fibonacci sequence up to n terms"""
        if n <= 0:
            return []
        elif n == 1:
            return [1]
        elif n == 2:
            return [1, 1]
        
        fib = [1, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib
    
    @staticmethod
    def calculate_growth_ratios(sequence: List[float]) -> List[float]:
        """Calculate growth ratios between consecutive terms"""
        if len(sequence) < 2:
            return []
        
        ratios = []
        for i in range(1, len(sequence)):
            if sequence[i-1] != 0:
                ratios.append(sequence[i] / sequence[i-1])
        return ratios
    
    @staticmethod
    def analyze_spiral_pattern(dots: List[Point]) -> Dict[str, float]:
        """Analyze spiral characteristics in dot arrangement"""
        if len(dots) < 3:
            return {"spiral_arms": 0, "growth_ratio": 1.0, "fibonacci_score": 0.0}
        
        # Find center point (centroid)
        center_x = sum(dot.x for dot in dots) / len(dots)
        center_y = sum(dot.y for dot in dots) / len(dots)
        center = Point(center_x, center_y)
        
        # Calculate distances and angles from center
        polar_coords = []
        for dot in dots:
            distance = center.distance_to(dot)
            angle = math.atan2(dot.y - center.y, dot.x - center.x)
            polar_coords.append((distance, angle))
        
        # Sort by angle to trace spiral
        polar_coords.sort(key=lambda x: x[1])
        
        # Analyze growth pattern
        distances = [coord[0] for coord in polar_coords]
        growth_ratios = FibonacciAnalyzer.calculate_growth_ratios(distances)
        
        # Compare with golden ratio (≈ 1.618)
        golden_ratio = (1 + math.sqrt(5)) / 2
        fibonacci_score = 0.0
        
        if growth_ratios:
            avg_ratio = sum(growth_ratios) / len(growth_ratios)
            # Score based on how close the average ratio is to golden ratio
            ratio_diff = abs(avg_ratio - golden_ratio)
            fibonacci_score = max(0, 1 - ratio_diff)
        
        # Estimate spiral arms (simplified)
        spiral_arms = len([r for r in growth_ratios if 1.2 < r < 2.0]) if growth_ratios else 1
        
        return {
            "spiral_arms": spiral_arms,
            "growth_ratio": sum(growth_ratios) / len(growth_ratios) if growth_ratios else 1.0,
            "fibonacci_score": fibonacci_score
        }

class GridClassifier:
    """Classifies different types of grid arrangements in Kolam patterns"""
    
    @staticmethod
    def analyze_neighbor_distances(dots: List[Point]) -> List[float]:
        """Calculate distances to nearest neighbors for each dot"""
        distances = []
        
        for i, dot1 in enumerate(dots):
            min_distance = float('inf')
            for j, dot2 in enumerate(dots):
                if i != j:
                    dist = dot1.distance_to(dot2)
                    if dist < min_distance:
                        min_distance = dist
            if min_distance != float('inf'):
                distances.append(min_distance)
        
        return distances
    
    @staticmethod
    def calculate_angle_distribution(dots: List[Point]) -> List[float]:
        """Calculate angle distribution to identify grid patterns"""
        if len(dots) < 3:
            return []
        
        angles = []
        center_x = sum(dot.x for dot in dots) / len(dots)
        center_y = sum(dot.y for dot in dots) / len(dots)
        
        for dot in dots:
            angle = math.atan2(dot.y - center_y, dot.x - center_x)
            angles.append(math.degrees(angle) % 360)
        
        return sorted(angles)
    
    @staticmethod
    def classify_grid_type(dots: List[Point]) -> GridType:
        """Classify the grid type based on dot arrangement"""
        if len(dots) < 4:
            return GridType.IRREGULAR
        
        # Analyze neighbor distances
        distances = GridClassifier.analyze_neighbor_distances(dots)
        
        if not distances:
            return GridType.IRREGULAR
        
        # Calculate distance uniformity
        avg_distance = sum(distances) / len(distances)
        distance_variance = sum((d - avg_distance)**2 for d in distances) / len(distances)
        uniformity = 1 / (1 + distance_variance/avg_distance**2)
        
        # Analyze angles
        angles = GridClassifier.calculate_angle_distribution(dots)
        
        # Simple heuristic classification
        if uniformity > 0.8:
            if len(angles) >= 6:
                # Check for hexagonal pattern (120° angles)
                angle_diffs = [(angles[i+1] - angles[i]) % 360 for i in range(len(angles)-1)]
                if any(abs(diff - 60) < 15 or abs(diff - 120) < 15 for diff in angle_diffs):
                    return GridType.HEXAGONAL
            
            # Check for square pattern (90° angles)
            if any(abs(angles[i] % 90) < 15 for i in range(len(angles))):
                return GridType.SQUARE
            
            # Check for triangular pattern
            angle_diffs = [(angles[i+1] - angles[i]) % 360 for i in range(len(angles)-1)]
            if any(abs(diff - 60) < 20 for diff in angle_diffs):
                return GridType.TRIANGULAR
        
        # Check for circular arrangement
        if len(dots) > 8:
            center_x = sum(dot.x for dot in dots) / len(dots)
            center_y = sum(dot.y for dot in dots) / len(dots)
            center = Point(center_x, center_y)
            
            radial_distances = [center.distance_to(dot) for dot in dots]
            avg_radius = sum(radial_distances) / len(radial_distances)
            radius_variance = sum((r - avg_radius)**2 for r in radial_distances) / len(radial_distances)
            
            if radius_variance / avg_radius**2 < 0.1:  # Low variance in radial distance
                return GridType.CIRCULAR
        
        return GridType.IRREGULAR

class ComplexityCalculator:
    """Calculates complexity scores for Kolam patterns"""
    
    @staticmethod
    def count_symmetry_axes(dots: List[Point], tolerance: float = 0.1) -> int:
        """Count number of symmetry axes in the pattern"""
        if len(dots) < 2:
            return 0
        
        center_x = sum(dot.x for dot in dots) / len(dots)
        center_y = sum(dot.y for dot in dots) / len(dots)
        
        symmetry_count = 0
        
        # Test various angles for reflection symmetry
        test_angles = [i * 15 for i in range(12)]  # Test every 15 degrees
        
        for angle_deg in test_angles:
            angle_rad = math.radians(angle_deg)
            is_symmetric = True
            
            for dot in dots:
                # Reflect point across line through center at given angle
                dx = dot.x - center_x
                dy = dot.y - center_y
                
                # Reflection formula
                cos_2a = math.cos(2 * angle_rad)
                sin_2a = math.sin(2 * angle_rad)
                
                reflected_x = center_x + dx * cos_2a + dy * sin_2a
                reflected_y = center_y + dx * sin_2a - dy * cos_2a
                
                # Check if reflected point exists in original set
                found_match = False
                for other_dot in dots:
                    if abs(other_dot.x - reflected_x) < tolerance and abs(other_dot.y - reflected_y) < tolerance:
                        found_match = True
                        break
                
                if not found_match:
                    is_symmetric = False
                    break
            
            if is_symmetric:
                symmetry_count += 1
        
        return symmetry_count
    
    @staticmethod
    def calculate_complexity_score(dots: List[Point], connections: List[Tuple[int, int]], 
                                 symmetry_axes: int) -> float:
        """Calculate overall complexity score for the pattern"""
        dot_count = len(dots)
        connection_count = len(connections)
        
        if dot_count == 0:
            return 0.0
        
        # Base complexity from dots and connections
        base_complexity = (dot_count + connection_count * 1.5) / 10
        
        # Symmetry bonus (more symmetry = more complex mathematical structure)
        symmetry_bonus = symmetry_axes * 0.2
        
        # Connection density factor
        max_possible_connections = dot_count * (dot_count - 1) // 2
        connection_density = connection_count / max_possible_connections if max_possible_connections > 0 else 0
        density_factor = 1 + connection_density
        
        final_score = (base_complexity + symmetry_bonus) * density_factor
        
        return min(final_score, 10.0)  # Cap at 10.0

# ================================
# MAIN ANALYSIS ENGINE
# ================================

class KolamMathematicalAnalyzer:
    """Main class that orchestrates the mathematical analysis of Kolam patterns"""
    
    def __init__(self):
        self.fibonacci_analyzer = FibonacciAnalyzer()
        self.grid_classifier = GridClassifier()
        self.complexity_calculator = ComplexityCalculator()
    
    def analyze_pattern(self, pattern: KolamPattern) -> AnalysisResult:
        """Perform complete mathematical analysis of a Kolam pattern"""
        
        # Fibonacci and spiral analysis
        spiral_analysis = self.fibonacci_analyzer.analyze_spiral_pattern(pattern.dots)
        
        # Grid classification
        grid_type = self.grid_classifier.classify_grid_type(pattern.dots)
        
        # Complexity analysis
        symmetry_axes = self.complexity_calculator.count_symmetry_axes(pattern.dots)
        complexity_score = self.complexity_calculator.calculate_complexity_score(
            pattern.dots, pattern.connections, symmetry_axes
        )
        
        # Generate educational explanation
        explanation = self._generate_explanation(
            pattern, grid_type, spiral_analysis, symmetry_axes, complexity_score
        )
        
        # Compile detailed metrics
        detailed_metrics = {
            "total_dots": len(pattern.dots),
            "total_connections": len(pattern.connections),
            "connection_density": len(pattern.connections) / max(1, len(pattern.dots) * (len(pattern.dots) - 1) // 2),
            "spiral_analysis": spiral_analysis,
            "grid_uniformity": self._calculate_grid_uniformity(pattern.dots),
            "pattern_bounds": self._calculate_pattern_bounds(pattern.dots)
        }
        
        return AnalysisResult(
            pattern_name=pattern.name,
            fibonacci_score=spiral_analysis["fibonacci_score"],
            grid_type=grid_type,
            complexity_score=complexity_score,
            symmetry_axes=symmetry_axes,
            spiral_arms=spiral_analysis["spiral_arms"],
            growth_ratio=spiral_analysis["growth_ratio"],
            educational_explanation=explanation,
            detailed_metrics=detailed_metrics
        )
    
    def _generate_explanation(self, pattern: KolamPattern, grid_type: GridType, 
                             spiral_analysis: Dict, symmetry_axes: int, 
                             complexity_score: float) -> str:
        """Generate educational explanation of the mathematical principles"""
        
        explanation = f"Mathematical Analysis of '{pattern.name}'\n"
        explanation += "=" * (len(explanation) - 1) + "\n\n"
        
        # Grid type explanation
        explanation += f"🔷 Grid Structure: {grid_type.value}\n"
        if grid_type == GridType.SQUARE:
            explanation += "This pattern follows a square grid arrangement, where dots are positioned at regular intervals forming right angles (90°). This creates a stable, balanced foundation often seen in architectural designs.\n\n"
        elif grid_type == GridType.HEXAGONAL:
            explanation += "This pattern exhibits a hexagonal grid structure with 120° angular relationships. Hexagonal patterns are highly efficient in nature and mathematics, offering maximum area coverage with minimum perimeter.\n\n"
        elif grid_type == GridType.TRIANGULAR:
            explanation += "The pattern displays triangular grid characteristics with 60° angular relationships. Triangular tessellations are fundamental in geometry and provide excellent structural stability.\n\n"
        elif grid_type == GridType.CIRCULAR:
            explanation += "This pattern arranges elements in a circular formation, representing unity and infinite continuity. Circular patterns often symbolize cosmic order and divine perfection in traditional designs.\n\n"
        
        # Fibonacci analysis
        explanation += f"🌀 Spiral Characteristics:\n"
        fibonacci_score = spiral_analysis["fibonacci_score"]
        if fibonacci_score > 0.7:
            explanation += f"Strong Fibonacci spiral detected! (Score: {fibonacci_score:.2f}/1.0)\n"
            explanation += f"The growth ratio ({spiral_analysis['growth_ratio']:.3f}) closely approximates the golden ratio (1.618), indicating natural mathematical harmony.\n\n"
        elif fibonacci_score > 0.3:
            explanation += f"Moderate spiral characteristics found (Score: {fibonacci_score:.2f}/1.0)\n"
            explanation += f"The pattern shows some spiral growth with a ratio of {spiral_analysis['growth_ratio']:.3f}.\n\n"
        else:
            explanation += f"Limited spiral characteristics (Score: {fibonacci_score:.2f}/1.0)\n"
            explanation += "The pattern is more linear or geometric rather than spiral in nature.\n\n"
        
        # Symmetry analysis
        explanation += f"⚖️ Symmetry Analysis:\n"
        if symmetry_axes >= 6:
            explanation += f"Highly symmetric pattern with {symmetry_axes} axes of symmetry! This indicates exceptional mathematical order and balance.\n\n"
        elif symmetry_axes >= 3:
            explanation += f"Good symmetry with {symmetry_axes} axes. This creates visual harmony and mathematical elegance.\n\n"
        elif symmetry_axes > 0:
            explanation += f"Basic symmetry present ({symmetry_axes} axes). The pattern shows some mathematical order.\n\n"
        else:
            explanation += "Asymmetric pattern. This may represent dynamic or organic growth rather than geometric order.\n\n"
        
        # Complexity assessment
        explanation += f"📊 Complexity Score: {complexity_score:.1f}/10.0\n"
        if complexity_score >= 7:
            explanation += "Highly complex pattern with intricate mathematical relationships. This demonstrates advanced geometric understanding.\n\n"
        elif complexity_score >= 4:
            explanation += "Moderately complex pattern with good mathematical structure. Shows clear geometric principles.\n\n"
        else:
            explanation += "Simple yet elegant pattern. Focus on fundamental geometric relationships.\n\n"
        
        # Educational insights
        explanation += "🎓 Mathematical Principles Demonstrated:\n"
        principles = []
        
        if fibonacci_score > 0.5:
            principles.append("• Fibonacci sequence and golden ratio relationships")
        if grid_type != GridType.IRREGULAR:
            principles.append(f"• {grid_type.value} tessellation patterns")
        if symmetry_axes > 0:
            principles.append("• Rotational and reflectional symmetry")
        if spiral_analysis["spiral_arms"] > 1:
            principles.append("• Spiral geometry and growth patterns")
        if len(pattern.connections) > 0:
            principles.append("• Graph theory and network connectivity")
        
        if principles:
            explanation += "\n".join(principles) + "\n\n"
        else:
            explanation += "• Basic geometric dot arrangements\n\n"
        
        explanation += "This analysis reveals the deep mathematical beauty embedded in traditional Kolam designs!"
        
        return explanation
    
    def _calculate_grid_uniformity(self, dots: List[Point]) -> float:
        """Calculate how uniform the grid spacing is"""
        if len(dots) < 2:
            return 0.0
        
        distances = self.grid_classifier.analyze_neighbor_distances(dots)
        if not distances:
            return 0.0
        
        avg_distance = sum(distances) / len(distances)
        variance = sum((d - avg_distance)**2 for d in distances) / len(distances)
        
        return 1 / (1 + variance/avg_distance**2)
    
    def _calculate_pattern_bounds(self, dots: List[Point]) -> Dict[str, float]:
        """Calculate the bounding box of the pattern"""
        if not dots:
            return {"width": 0, "height": 0, "area": 0}
        
        min_x = min(dot.x for dot in dots)
        max_x = max(dot.x for dot in dots)
        min_y = min(dot.y for dot in dots)
        max_y = max(dot.y for dot in dots)
        
        width = max_x - min_x
        height = max_y - min_y
        
        return {
            "width": width,
            "height": height,
            "area": width * height
        }

# ================================
# PATTERN GENERATORS & USER INPUT
# ================================

class PatternGenerator:
    """Generates various types of Kolam patterns for testing and demonstration"""
    
    @staticmethod
    def create_square_grid_pattern(name: str, rows: int = 5, cols: int = 5, spacing: float = 1.0) -> KolamPattern:
        """Generate a square grid pattern"""
        dots = []
        for i in range(rows):
            for j in range(cols):
                dots.append(Point(j * spacing, i * spacing))
        
        # Create connections between adjacent dots
        connections = []
        for i in range(rows):
            for j in range(cols):
                current_idx = i * cols + j
                # Connect to right neighbor
                if j < cols - 1:
                    connections.append((current_idx, current_idx + 1))
                # Connect to bottom neighbor
                if i < rows - 1:
                    connections.append((current_idx, current_idx + cols))
        
        return KolamPattern(name, dots, connections, GridType.SQUARE)
    
    @staticmethod
    def create_circular_pattern(name: str, radius: float = 3.0, num_points: int = 12) -> KolamPattern:
        """Generate a circular pattern"""
        dots = []
        center = Point(0, 0)
        dots.append(center)  # Add center dot
        
        # Add dots in a circle
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            dots.append(Point(x, y))
        
        # Connect center to all outer dots
        connections = [(0, i) for i in range(1, num_points + 1)]
        
        # Connect adjacent outer dots
        for i in range(1, num_points):
            connections.append((i, i + 1))
        connections.append((num_points, 1))  # Close the circle
        
        return KolamPattern(name, dots, connections, GridType.CIRCULAR)
    
    @staticmethod
    def create_fibonacci_spiral_pattern(name: str, turns: int = 3) -> KolamPattern:
        """Generate a pattern based on Fibonacci spiral"""
        dots = []
        connections = []
        
        fibonacci_seq = [1, 1, 2, 3, 5, 8, 13, 21]
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        # Create spiral based on Fibonacci numbers
        for i in range(min(turns * 4, len(fibonacci_seq))):
            angle = i * math.pi / 2  # 90 degrees per turn
            radius = fibonacci_seq[i] * 0.5
            
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            dots.append(Point(x, y))
            
            # Connect to previous dot
            if i > 0:
                connections.append((i-1, i))
        
        return KolamPattern(name, dots, connections, GridType.IRREGULAR)
    
    @staticmethod
    def create_hexagonal_pattern(name: str, layers: int = 3) -> KolamPattern:
        """Generate a hexagonal pattern"""
        dots = []
        connections = []
        
        # Center dot
        dots.append(Point(0, 0))
        
        # Add hexagonal layers
        for layer in range(1, layers + 1):
            for i in range(6):  # 6 sides of hexagon
                for j in range(layer):
                    angle = i * math.pi / 3 + (j / layer) * (math.pi / 3)
                    x = layer * math.cos(angle)
                    y = layer * math.sin(angle)
                    dots.append(Point(x, y))
        
        # Simple connections (this could be made more sophisticated)
        for i in range(len(dots) - 1):
            if i < len(dots) // 2:
                connections.append((i, i + 1))
        
        return KolamPattern(name, dots, connections, GridType.HEXAGONAL)

def get_user_input_pattern() -> KolamPattern:
    """Interactive function to get pattern input from user"""
    print("🎨 Welcome to Kolam Mathematical Principle Identifier!")
    print("=" * 55)
    print("Choose how you'd like to input your Kolam pattern:\n")
    
    print("1. 📐 Generate a test pattern (recommended for beginners)")
    print("2. ✏️  Create custom pattern by entering coordinates")
    print("3. 🎲 Generate random pattern")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                return get_test_pattern()
            elif choice == "2":
                return get_custom_pattern()
            elif choice == "3":
                return get_random_pattern()
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            exit()
        except Exception as e:
            print(f"❌ Error: {e}. Please try again.")

def get_test_pattern() -> KolamPattern:
    """Let user choose from predefined test patterns"""
    print("\n📚 Available Test Patterns:")
    print("1. Square Grid (5x5)")
    print("2. Circular Pattern (12 points)")
    print("3. Fibonacci Spiral")
    print("4. Hexagonal Pattern")
    print("5. Simple Triangle")
    
    while True:
        try:
            choice = input("\nSelect a test pattern (1-5): ").strip()
            generator = PatternGenerator()
            
            if choice == "1":
                return generator.create_square_grid_pattern("Square Grid 5x5")
            elif choice == "2":
                return generator.create_circular_pattern("12-Point Circle")
            elif choice == "3":
                return generator.create_fibonacci_spiral_pattern("Fibonacci Spiral")
            elif choice == "4":
                return generator.create_hexagonal_pattern("Hexagonal Pattern")
            elif choice == "5":
                # Create a simple triangle
                dots = [Point(0, 0), Point(2, 0), Point(1, math.sqrt(3))]
                connections = [(0, 1), (1, 2), (2, 0)]
                return KolamPattern("Simple Triangle", dots, connections)
            else:
                print("❌ Invalid choice. Please enter 1-5.")
        except Exception as e:
            print(f"❌ Error: {e}. Please try again.")

def get_custom_pattern() -> KolamPattern:
    """Let user create a custom pattern by entering coordinates"""
    print("\n✏️  Custom Pattern Creation")
    print("Enter dot coordinates one by one. Type 'done' when finished.")
    print("Format: x,y (example: 1,2 or 3.5,4.2)")
    
    dots = []
    
    while True:
        try:
            user_input = input(f"Dot {len(dots)+1} coordinates (or 'done'): ").strip().lower()
            
            if user_input == 'done':
                if len(dots) < 2:
                    print("❌ Need at least 2 dots to create a pattern.")
                    continue
                break
            
            # Parse coordinates
            coords = user_input.split(',')
            if len(coords) != 2:
                print("❌ Invalid format. Use: x,y")
                continue
            
            x = float(coords[0].strip())
            y = float(coords[1].strip())
            dots.append(Point(x, y))
            print(f"✅ Added dot at ({x}, {y})")
            
        except ValueError:
            print("❌ Invalid coordinates. Please enter numbers.")
        except KeyboardInterrupt:
            print("\n👋 Cancelled.")
            return get_user_input_pattern()
    
    # Get pattern name
    name = input("\nEnter a name for your pattern: ").strip()
    if not name:
        name = f"Custom Pattern ({len(dots)} dots)"
    
    # Ask about connections
    connections = []
    print(f"\n🔗 Connection Setup (optional)")
    print("Connect dots by specifying pairs of dot numbers (1-indexed)")
    print("Example: 1,2 connects the first and second dots")
    print("Type 'auto' for automatic connections or 'done' to skip")
    
    while True:
        try:
            conn_input = input("Connection (dot1,dot2) or 'auto'/'done': ").strip().lower()
            
            if conn_input == 'done':
                break
            elif conn_input == 'auto':
                # Auto-connect adjacent dots in order
                for i in range(len(dots) - 1):
                    connections.append((i, i + 1))
                print(f"✅ Added {len(connections)} automatic connections")
                break
            
            # Parse connection
            conn_parts = conn_input.split(',')
            if len(conn_parts) != 2:
                print("❌ Invalid format. Use: dot1,dot2")
                continue
            
            dot1 = int(conn_parts[0].strip()) - 1  # Convert to 0-indexed
            dot2 = int(conn_parts[1].strip()) - 1
            
            if 0 <= dot1 < len(dots) and 0 <= dot2 < len(dots) and dot1 != dot2:
                connections.append((dot1, dot2))
                print(f"✅ Connected dots {dot1+1} and {dot2+1}")
            else:
                print(f"❌ Invalid dot numbers. Use 1-{len(dots)}")
                
        except ValueError:
            print("❌ Invalid input. Please enter numbers.")
        except KeyboardInterrupt:
            print("\n👋 Cancelled.")
            break
    
    return KolamPattern(name, dots, connections)

def get_random_pattern() -> KolamPattern:
    """Generate a random pattern"""
    import random
    
    print("\n🎲 Random Pattern Generation")
    
    # Get parameters
    while True:
        try:
            num_dots = int(input("Number of dots (5-20): ").strip())
            if 5 <= num_dots <= 20:
                break
            print("❌ Please enter a number between 5 and 20.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Generate random dots
    dots = []
    random.seed()  # Use current time as seed
    
    for i in range(num_dots):
        x = random.uniform(-5, 5)
        y = random.uniform(-5, 5)
        dots.append(Point(x, y))
    
    # Generate some random connections
    connections = []
    num_connections = random.randint(num_dots//2, num_dots * 2)
    
    for _ in range(num_connections):
        dot1 = random.randint(0, num_dots - 1)
        dot2 = random.randint(0, num_dots - 1)
        
        if dot1 != dot2 and (dot1, dot2) not in connections and (dot2, dot1) not in connections:
            connections.append((dot1, dot2))
    
    name = f"Random Pattern ({num_dots} dots, {len(connections)} connections)"
    return KolamPattern(name, dots, connections)

# ================================
# VISUALIZATION TOOLS
# ================================

class KolamVisualizer:
    """Handles visualization of Kolam patterns and analysis results"""
    
    @staticmethod
    def plot_pattern(pattern: KolamPattern, analysis: AnalysisResult = None, figsize: Tuple[int, int] = (12, 8)):
        """Create a comprehensive visualization of the Kolam pattern"""
        
        if analysis:
            # Create subplots for detailed analysis
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Mathematical Analysis of {pattern.name}', fontsize=16, fontweight='bold')
        else:
            # Single plot
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
            ax2 = ax3 = ax4 = None
        
        # Main pattern plot
        KolamVisualizer._plot_main_pattern(ax1, pattern, analysis)
        
        if analysis and ax2 and ax3 and ax4:
            # Additional analysis plots
            KolamVisualizer._plot_fibonacci_analysis(ax2, pattern, analysis)
            KolamVisualizer._plot_symmetry_analysis(ax3, pattern, analysis)
            KolamVisualizer._plot_metrics_summary(ax4, analysis)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def _plot_main_pattern(ax, pattern: KolamPattern, analysis: AnalysisResult = None):
        """Plot the main pattern with dots and connections"""
        
        # Extract coordinates
        x_coords = [dot.x for dot in pattern.dots]
        y_coords = [dot.y for dot in pattern.dots]
        
        # Plot connections first (so they appear behind dots)
        for connection in pattern.connections:
            if 0 <= connection[0] < len(pattern.dots) and 0 <= connection[1] < len(pattern.dots):
                x1, y1 = pattern.dots[connection[0]].x, pattern.dots[connection[0]].y
                x2, y2 = pattern.dots[connection[1]].x, pattern.dots[connection[1]].y
                ax.plot([x1, x2], [y1, y2], 'b-', alpha=0.6, linewidth=1.5)
        
        # Plot dots
        ax.scatter(x_coords, y_coords, c='red', s=100, zorder=5, edgecolors='darkred', linewidth=2)
        
        # Add dot numbers
        for i, dot in enumerate(pattern.dots):
            ax.annotate(str(i+1), (dot.x, dot.y), xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, fontweight='bold', color='white')
        
        # Formatting
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{pattern.name}\n({len(pattern.dots)} dots, {len(pattern.connections)} connections)', 
                    fontsize=12, fontweight='bold')
        
        if analysis:
            # Add analysis info as text
            info_text = f"Grid Type: {analysis.grid_type.value}\n"
            info_text += f"Complexity: {analysis.complexity_score:.1f}/10.0\n"
            info_text += f"Fibonacci Score: {analysis.fibonacci_score:.2f}\n"
            info_text += f"Symmetry Axes: {analysis.symmetry_axes}"
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    @staticmethod
    def _plot_fibonacci_analysis(ax, pattern: KolamPattern, analysis: AnalysisResult):
        """Plot Fibonacci/spiral analysis"""
        
        # Calculate center
        center_x = sum(dot.x for dot in pattern.dots) / len(pattern.dots)
        center_y = sum(dot.y for dot in pattern.dots) / len(pattern.dots)
        
        # Calculate polar coordinates
        distances = []
        angles = []
        
        for dot in pattern.dots:
            dist = math.sqrt((dot.x - center_x)**2 + (dot.y - center_y)**2)
            angle = math.atan2(dot.y - center_y, dot.x - center_x)
            distances.append(dist)
            angles.append(angle)
        
        # Sort by angle
        sorted_data = sorted(zip(angles, distances))
        sorted_angles, sorted_distances = zip(*sorted_data) if sorted_data else ([], [])
        
        # Plot spiral growth
        if len(sorted_distances) > 1:
            ax.plot(range(len(sorted_distances)), sorted_distances, 'bo-', markersize=6)
            ax.set_xlabel('Point Order (by angle)')
            ax.set_ylabel('Distance from Center')
            ax.set_title(f'Spiral Growth Pattern\n(Fibonacci Score: {analysis.fibonacci_score:.2f})')
            ax.grid(True, alpha=0.3)
            
            # Add golden ratio reference line if applicable
            if len(sorted_distances) > 2:
                golden_ratio = (1 + math.sqrt(5)) / 2
                theoretical = [sorted_distances[0] * (golden_ratio ** i) for i in range(len(sorted_distances))]
                ax.plot(range(len(theoretical)), theoretical, 'r--', alpha=0.7, label='Golden Ratio Growth')
                ax.legend()
        else:
            ax.text(0.5, 0.5, 'Insufficient data\nfor spiral analysis', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
    
    @staticmethod
    def _plot_symmetry_analysis(ax, pattern: KolamPattern, analysis: AnalysisResult):
        """Plot symmetry analysis"""
        
        # Create a polar-like plot showing symmetry axes
        angles = np.linspace(0, 2*np.pi, 360)
        
        # Plot the pattern in polar context
        center_x = sum(dot.x for dot in pattern.dots) / len(pattern.dots)
        center_y = sum(dot.y for dot in pattern.dots) / len(pattern.dots)
        
        # Convert dots to polar coordinates
        polar_angles = []
        polar_radii = []
        
        for dot in pattern.dots:
            angle = math.atan2(dot.y - center_y, dot.x - center_x)
            radius = math.sqrt((dot.x - center_x)**2 + (dot.y - center_y)**2)
            polar_angles.append(angle)
            polar_radii.append(radius)
        
        # Create polar plot
        ax = plt.subplot(2, 2, 3, projection='polar')
        
        # Plot dots
        ax.scatter(polar_angles, polar_radii, c='red', s=60, alpha=0.8)
        
        # Highlight symmetry axes
        if analysis.symmetry_axes > 0:
            for i in range(analysis.symmetry_axes):
                axis_angle = i * 2 * math.pi / max(analysis.symmetry_axes, 1)
                max_radius = max(polar_radii) if polar_radii else 1
                ax.plot([axis_angle, axis_angle], [0, max_radius], 'g--', alpha=0.7, linewidth=2)
        
        ax.set_title(f'Symmetry Analysis\n({analysis.symmetry_axes} axes detected)', fontsize=11)
    
    @staticmethod
    def _plot_metrics_summary(ax, analysis: AnalysisResult):
        """Plot a summary of key metrics"""
        
        # Prepare data for bar chart
        metrics = {
            'Fibonacci\nScore': analysis.fibonacci_score,
            'Complexity\n(scaled)': analysis.complexity_score / 10,
            'Symmetry\n(scaled)': min(analysis.symmetry_axes / 8, 1.0),
            'Spiral Arms\n(scaled)': min(analysis.spiral_arms / 5, 1.0)
        }
        
        # Create bar chart
        bars = ax.bar(metrics.keys(), metrics.values(), 
                     color=['gold', 'skyblue', 'lightgreen', 'salmon'])
        
        # Add value labels on bars
        for bar, (key, value) in zip(bars, metrics.items()):
            height = bar.get_height()
            if 'scaled' in key:
                original_value = value * (10 if 'Complexity' in key else 8 if 'Symmetry' in key else 5)
                label = f'{original_value:.1f}'
            else:
                label = f'{value:.2f}'
            
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   label, ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Normalized Score')
        ax.set_title('Metrics Summary', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# ================================
# INTERACTIVE INTERFACE
# ================================

def print_detailed_analysis(analysis: AnalysisResult):
    """Print detailed analysis results in a formatted way"""
    
    print("\n" + "="*80)
    print(f"🔍 DETAILED MATHEMATICAL ANALYSIS")
    print("="*80)
    
    print(f"\n📋 Pattern Name: {analysis.pattern_name}")
    print(f"🏗️  Grid Type: {analysis.grid_type.value}")
    print(f"⭐ Overall Complexity: {analysis.complexity_score:.2f}/10.0")
    
    print(f"\n🌀 SPIRAL & FIBONACCI ANALYSIS:")
    print(f"   • Fibonacci Score: {analysis.fibonacci_score:.3f}/1.000")
    print(f"   • Growth Ratio: {analysis.growth_ratio:.3f} (Golden Ratio ≈ 1.618)")
    print(f"   • Spiral Arms Detected: {analysis.spiral_arms}")
    
    print(f"\n⚖️ SYMMETRY ANALYSIS:")
    print(f"   • Symmetry Axes: {analysis.symmetry_axes}")
    if analysis.symmetry_axes >= 6:
        symmetry_level = "Exceptionally High"
    elif analysis.symmetry_axes >= 3:
        symmetry_level = "High"
    elif analysis.symmetry_axes > 0:
        symmetry_level = "Moderate"
    else:
        symmetry_level = "Low/Asymmetric"
    print(f"   • Symmetry Level: {symmetry_level}")
    
    print(f"\n📊 DETAILED METRICS:")
    metrics = analysis.detailed_metrics
    print(f"   • Total Dots: {metrics['total_dots']}")
    print(f"   • Total Connections: {metrics['total_connections']}")
    print(f"   • Connection Density: {metrics['connection_density']:.3f}")
    print(f"   • Grid Uniformity: {metrics['grid_uniformity']:.3f}")
    print(f"   • Pattern Width: {metrics['pattern_bounds']['width']:.2f}")
    print(f"   • Pattern Height: {metrics['pattern_bounds']['height']:.2f}")
    print(f"   • Pattern Area: {metrics['pattern_bounds']['area']:.2f}")
    
    print(f"\n🎓 EDUCATIONAL EXPLANATION:")
    print("-" * 50)
    print(analysis.educational_explanation)
    
    print("\n" + "="*80)

def run_interactive_analysis():
    """Main interactive function to run the analysis"""
    
    print("🕉️  KOLAM MATHEMATICAL PRINCIPLE IDENTIFIER 🕉️")
    print("Enhanced MVP 2 - Interactive Analysis Tool")
    print("=" * 60)
    
    try:
        # Get pattern from user
        pattern = get_user_input_pattern()
        
        print(f"\n✨ Analyzing pattern: '{pattern.name}'...")
        print("🔄 Processing mathematical principles...")
        
        # Create analyzer and run analysis
        analyzer = KolamMathematicalAnalyzer()
        analysis = analyzer.analyze_pattern(pattern)
        
        # Display results
        print_detailed_analysis(analysis)
        
        # Ask if user wants visualization
        while True:
            try:
                show_viz = input("\n📊 Would you like to see the visual analysis? (y/n): ").strip().lower()
                if show_viz in ['y', 'yes', '1', 'true']:
                    print("🎨 Generating visualizations...")
                    
                    # Create visualization
                    fig = KolamVisualizer.plot_pattern(pattern, analysis)
                    plt.show()
                    break
                elif show_viz in ['n', 'no', '0', 'false']:
                    break
                else:
                    print("❌ Please enter 'y' or 'n'")
            except KeyboardInterrupt:
                break
        
        # Ask if user wants to analyze another pattern
        while True:
            try:
                another = input("\n🔄 Would you like to analyze another pattern? (y/n): ").strip().lower()
                if another in ['y', 'yes', '1', 'true']:
                    print("\n" + "="*60)
                    run_interactive_analysis()
                    break
                elif another in ['n', 'no', '0', 'false']:
                    print("\n🙏 Thank you for using the Kolam Mathematical Principle Identifier!")
                    print("✨ Keep exploring the beauty of mathematics in traditional art! ✨")
                    break
                else:
                    print("❌ Please enter 'y' or 'n'")
            except KeyboardInterrupt:
                break
    
    except KeyboardInterrupt:
        print("\n\n👋 Analysis cancelled. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please try again or contact support.")

# ================================
# EXAMPLE USAGE & TESTING
# ================================

def run_example_demonstrations():
    """Run demonstrations with various pattern types"""
    
    print("🎭 DEMONSTRATION MODE - Testing Various Pattern Types")
    print("=" * 60)
    
    analyzer = KolamMathematicalAnalyzer()
    generator = PatternGenerator()
    
    # Test patterns
    test_patterns = [
        generator.create_square_grid_pattern("Demo Square Grid", 4, 4),
        generator.create_circular_pattern("Demo Circle", 2.5, 8),
        generator.create_fibonacci_spiral_pattern("Demo Fibonacci Spiral", 2),
        generator.create_hexagonal_pattern("Demo Hexagon", 2)
    ]
    
    for i, pattern in enumerate(test_patterns, 1):
        print(f"\n🔍 Analysis {i}/4: {pattern.name}")
        print("-" * 40)
        
        analysis = analyzer.analyze_pattern(pattern)
        
        # Quick summary
        print(f"Grid Type: {analysis.grid_type.value}")
        print(f"Complexity: {analysis.complexity_score:.1f}/10.0")
        print(f"Fibonacci Score: {analysis.fibonacci_score:.2f}")
        print(f"Symmetry Axes: {analysis.symmetry_axes}")
        
        if i < len(test_patterns):
            input("\nPress Enter to continue to next pattern...")
    
    print(f"\n✅ Demonstration complete!")
    print("🎯 Ready for interactive use with run_interactive_analysis()")

# ================================
# COLAB-SPECIFIC HELPERS
# ================================

def setup_colab_environment():
    """Setup function specifically for Google Colab"""
    try:
        import google.colab
        print("🔧 Google Colab environment detected!")
        print("📦 All required packages are included in the script.")
        return True
    except ImportError:
        print("💻 Running in local environment")
        return False

def display_usage_instructions():
    """Display comprehensive usage instructions"""
    
    instructions = """
🎯 KOLAM MATHEMATICAL PRINCIPLE IDENTIFIER - USAGE GUIDE
========================================================

This tool analyzes mathematical principles in Kolam patterns including:
• Fibonacci sequences and golden ratio relationships
• Grid type classification (square, hexagonal, triangular, circular)
• Symmetry analysis with axis detection
• Complexity scoring based on mathematical relationships

📚 MAIN FUNCTIONS:
-----------------

1. run_interactive_analysis()
   → Start the main interactive analysis tool
   → Choose from test patterns or create your own
   → Get detailed mathematical analysis and visualizations

2. run_example_demonstrations() 
   → See quick demos of different pattern types
   → Great for understanding the tool's capabilities

3. Manual Analysis (Advanced):
   → Create KolamPattern objects directly
   → Use KolamMathematicalAnalyzer for custom analysis
   → Full programmatic control

🚀 QUICK START:
--------------
Just run: run_interactive_analysis()

📊 ANALYSIS OUTPUT:
------------------
• Fibonacci Score: How closely the pattern follows Fibonacci spiral growth
• Grid Classification: Identifies the underlying geometric structure  
• Complexity Score: Overall mathematical complexity (1-10 scale)
• Symmetry Analysis: Number and types of symmetry axes
• Educational Explanations: Human-readable mathematical insights

🎨 VISUALIZATIONS:
-----------------
• Main pattern plot with dots and connections
• Spiral growth analysis charts
• Symmetry axis visualization
• Comprehensive metrics summary

💡 TIPS:
-------
• Start with test patterns to understand the analysis
• Try different grid types to see classification in action
• Experiment with symmetrical vs asymmetrical patterns
• The tool works best with 5-20 dots for clear analysis

🔬 MATHEMATICAL PRINCIPLES DETECTED:
----------------------------------
✓ Fibonacci sequences and golden ratio
✓ Regular tessellation patterns
✓ Rotational and reflectional symmetry
✓ Spiral geometry and growth patterns
✓ Graph theory relationships
✓ Geometric proportions and ratios

Ready to explore the mathematical beauty of Kolam patterns! 🕉️
"""
    
    print(instructions)

# ================================
# MAIN EXECUTION
# ================================

if __name__ == "__main__":
    # Initialize environment
    setup_colab_environment()
    
    # Display instructions
    display_usage_instructions()
    
    # Ready for use
    print("\n🎯 READY FOR USE!")
    print("🚀 Run: run_interactive_analysis() to start")
    print("🎭 Or: run_example_demonstrations() for quick demos")
