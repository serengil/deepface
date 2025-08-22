#!/usr/bin/env python3
"""
DeepFace Model Tester
Tests different DeepFace models with test images to find the best performing model.
"""

import os
import cv2
import numpy as np
from deepface import DeepFace
import time
from typing import Dict, List, Tuple
import json

class DeepFaceModelTester:
    def __init__(self, test_images_dir: str = "test_images"):
        """
        Initialize the model tester.
        
        Args:
            test_images_dir: Directory containing test images
        """
        self.test_images_dir = test_images_dir
        self.models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepID", "ArcFace", "SFace"]
        self.distance_metrics = ["cosine", "euclidean", "euclidean_l2"]
        self.thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.results = {}
        
    def load_test_images(self) -> Dict[str, str]:
        """
        Load test images from the test_images directory.
        Expected structure:
        test_images/
        ├── same_person_1.jpg
        ├── same_person_2.jpg
        └── different_person.jpg
        
        Returns:
            Dictionary with image paths
        """
        images = {}
        if not os.path.exists(self.test_images_dir):
            print(f"❌ Test images directory '{self.test_images_dir}' not found!")
            print("Please create the directory and add your test images:")
            print("  - same_person_1.jpg (first image of same person)")
            print("  - same_person_2.jpg (second image of same person)")
            print("  - different_person.jpg (image of different person)")
            return images
            
        expected_files = ["same_person_1.jpg", "same_person_2.jpg", "different_person.jpg"]
        # Also check for webp files
        webp_files = ["same_person_1.webp", "same_person_2.webp", "different_person.webp"]
        
        # Check for jpg files first
        for file in expected_files:
            file_path = os.path.join(self.test_images_dir, file)
            if os.path.exists(file_path):
                images[file.replace(".jpg", "")] = file_path
            else:
                # Check for webp version
                webp_file = file.replace(".jpg", ".webp")
                webp_path = os.path.join(self.test_images_dir, webp_file)
                if os.path.exists(webp_path):
                    images[file.replace(".jpg", "")] = webp_path
                else:
                    print(f"⚠️  Missing test image: {file} or {webp_file}")
                
        return images
    
    def test_model_combination(self, model: str, distance_metric: str, threshold: float, 
                             same_person_1: str, same_person_2: str, different_person: str) -> Dict:
        """
        Test a specific model combination with the test images.
        
        Returns:
            Dictionary with test results
        """
        results = {
            "model": model,
            "distance_metric": distance_metric,
            "threshold": threshold,
            "same_person_test": None,
            "different_person_test": None,
            "accuracy": 0.0,
            "avg_time": 0.0,
            "errors": []
        }
        
        test_cases = [
            ("same_person", same_person_1, same_person_2, True),
            ("different_person", same_person_1, different_person, False)
        ]
        
        times = []
        
        for test_name, img1_path, img2_path, expected_result in test_cases:
            try:
                start_time = time.time()
                
                result = DeepFace.verify(
                    img1_path=img1_path,
                    img2_path=img2_path,
                    model_name=model,
                    distance_metric=distance_metric,
                    enforce_detection=False
                )
                
                end_time = time.time()
                times.append(end_time - start_time)
                
                verified = result.get("verified", False)
                distance = result.get("distance", 0)
                
                # Check if result matches expected
                correct = (verified == expected_result)
                
                if test_name == "same_person":
                    results["same_person_test"] = {
                        "verified": verified,
                        "expected": expected_result,
                        "distance": distance,
                        "correct": correct
                    }
                else:
                    results["different_person_test"] = {
                        "verified": verified,
                        "expected": expected_result,
                        "distance": distance,
                        "correct": correct
                    }
                    
            except Exception as e:
                results["errors"].append(f"{test_name}: {str(e)}")
                times.append(0)
        
        # Calculate accuracy and average time
        correct_tests = 0
        if results["same_person_test"] and results["same_person_test"]["correct"]:
            correct_tests += 1
        if results["different_person_test"] and results["different_person_test"]["correct"]:
            correct_tests += 1
            
        results["accuracy"] = (correct_tests / 2) * 100
        results["avg_time"] = np.mean(times) if times else 0
        
        return results
    
    def run_comprehensive_test(self) -> Dict:
        """
        Run comprehensive tests across all model combinations.
        
        Returns:
            Dictionary with all test results
        """
        print("🔍 Loading test images...")
        images = self.load_test_images()
        
        if not images or len(images) < 3:
            print("❌ Not enough test images found. Please add the required test images.")
            return {}
        
        print(f"✅ Found {len(images)} test images")
        print(f"📊 Testing {len(self.models)} models with {len(self.distance_metrics)} distance metrics and {len(self.thresholds)} thresholds")
        print("=" * 80)
        
        all_results = []
        total_combinations = len(self.models) * len(self.distance_metrics) * len(self.thresholds)
        current_combination = 0
        
        for model in self.models:
            for distance_metric in self.distance_metrics:
                for threshold in self.thresholds:
                    current_combination += 1
                    print(f"Testing {current_combination}/{total_combinations}: {model} + {distance_metric} + {threshold}")
                    
                    result = self.test_model_combination(
                        model, distance_metric, threshold,
                        images["same_person_1"], images["same_person_2"], images["different_person"]
                    )
                    
                    all_results.append(result)
                    
                    # Print immediate results for good combinations
                    if result["accuracy"] == 100.0:
                        print(f"  🎯 PERFECT MATCH: {model} + {distance_metric} + {threshold:.1f}")
                    elif result["accuracy"] >= 50.0:
                        print(f"  ✅ Good: {result['accuracy']:.1f}% accuracy")
        
        return all_results
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """
        Analyze test results and find the best performing combinations.
        
        Returns:
            Dictionary with analysis results
        """
        if not results:
            return {}
        
        # Filter out results with errors
        valid_results = [r for r in results if not r["errors"]]
        
        if not valid_results:
            print("❌ No valid test results found!")
            return {}
        
        # Find best combinations by accuracy
        perfect_matches = [r for r in valid_results if r["accuracy"] == 100.0]
        good_matches = [r for r in valid_results if r["accuracy"] >= 50.0]
        
        # Sort by accuracy, then by speed
        valid_results.sort(key=lambda x: (x["accuracy"], -x["avg_time"]), reverse=True)
        
        analysis = {
            "total_tests": len(results),
            "valid_tests": len(valid_results),
            "perfect_matches": len(perfect_matches),
            "good_matches": len(good_matches),
            "best_combinations": valid_results[:10],  # Top 10
            "perfect_combinations": perfect_matches,
            "fastest_perfect": None,
            "most_accurate": valid_results[0] if valid_results else None
        }
        
        # Find fastest perfect match
        if perfect_matches:
            fastest_perfect = min(perfect_matches, key=lambda x: x["avg_time"])
            analysis["fastest_perfect"] = fastest_perfect
        
        return analysis
    
    def print_analysis(self, analysis: Dict):
        """
        Print the analysis results in a formatted way.
        """
        print("\n" + "=" * 80)
        print("📊 TEST RESULTS ANALYSIS")
        print("=" * 80)
        
        print(f"Total tests run: {analysis['total_tests']}")
        print(f"Valid tests: {analysis['valid_tests']}")
        print(f"Perfect matches (100% accuracy): {analysis['perfect_matches']}")
        print(f"Good matches (≥50% accuracy): {analysis['good_matches']}")
        
        if analysis["most_accurate"]:
            best = analysis["most_accurate"]
            print(f"\n🏆 BEST OVERALL PERFORMANCE:")
            print(f"  Model: {best['model']}")
            print(f"  Distance Metric: {best['distance_metric']}")
            print(f"  Threshold: {best['threshold']}")
            print(f"  Accuracy: {best['accuracy']:.1f}%")
            print(f"  Average Time: {best['avg_time']:.3f}s")
            
            if best["same_person_test"]:
                sp = best["same_person_test"]
                print(f"  Same Person Test: {'✅' if sp['correct'] else '❌'} (Distance: {sp['distance']:.4f})")
            if best["different_person_test"]:
                dp = best["different_person_test"]
                print(f"  Different Person Test: {'✅' if dp['correct'] else '❌'} (Distance: {dp['distance']:.4f})")
        
        if analysis["fastest_perfect"]:
            fastest = analysis["fastest_perfect"]
            print(f"\n⚡ FASTEST PERFECT MATCH:")
            print(f"  Model: {fastest['model']}")
            print(f"  Distance Metric: {fastest['distance_metric']}")
            print(f"  Threshold: {fastest['threshold']}")
            print(f"  Average Time: {fastest['avg_time']:.3f}s")
        
        if analysis["perfect_combinations"]:
            print(f"\n🎯 ALL PERFECT COMBINATIONS ({len(analysis['perfect_combinations'])} found):")
            for i, combo in enumerate(analysis["perfect_combinations"][:5], 1):  # Show top 5
                print(f"  {i}. {combo['model']} + {combo['distance_metric']} + {combo['threshold']:.1f} ({combo['avg_time']:.3f}s)")
        
        print(f"\n📋 TOP 10 COMBINATIONS:")
        for i, combo in enumerate(analysis["best_combinations"][:10], 1):
            print(f"  {i}. {combo['model']} + {combo['distance_metric']} + {combo['threshold']:.1f} - {combo['accuracy']:.1f}% ({combo['avg_time']:.3f}s)")
    
    def save_results(self, results: List[Dict], analysis: Dict, filename: str = "model_test_results.json"):
        """
        Save test results to a JSON file.
        """
        output = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_images_dir": self.test_images_dir,
            "models_tested": self.models,
            "distance_metrics_tested": self.distance_metrics,
            "thresholds_tested": self.thresholds,
            "all_results": results,
            "analysis": analysis
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
    
    def generate_recommendation(self, analysis: Dict) -> str:
        """
        Generate a recommendation for the best model to use.
        """
        if not analysis["most_accurate"]:
            return "No valid model combinations found."
        
        best = analysis["most_accurate"]
        fastest_perfect = analysis["fastest_perfect"]
        
        recommendation = f"""
🎯 RECOMMENDATION FOR YOUR AI IMAGE EDITOR:

BEST OVERALL: {best['model']} + {best['distance_metric']} + {best['threshold']:.1f}
- Accuracy: {best['accuracy']:.1f}%
- Speed: {best['avg_time']:.3f}s per comparison

"""
        
        if fastest_perfect and fastest_perfect != best:
            recommendation += f"FASTEST PERFECT: {fastest_perfect['model']} + {fastest_perfect['distance_metric']} + {fastest_perfect['threshold']:.1f}\n"
            recommendation += f"- Speed: {fastest_perfect['avg_time']:.3f}s per comparison\n\n"
        
        recommendation += f"""
📝 TO UPDATE YOUR EDITOR.PY:

For Photo Upload verification, use:
MODEL_NAME = "{best['model']}"
THRESHOLD = {best['threshold']:.1f}
distance_metric="{best['distance_metric']}"

This combination achieved {best['accuracy']:.1f}% accuracy on your test images.
"""
        
        return recommendation

def main():
    """
    Main function to run the model tester.
    """
    print("🤖 DeepFace Model Tester")
    print("=" * 50)
    
    # Create test images directory if it doesn't exist
    test_dir = "test_images"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"📁 Created test images directory: {test_dir}")
        print("Please add your test images:")
        print("  - same_person_1.jpg (first image of same person)")
        print("  - same_person_2.jpg (second image of same person)")
        print("  - different_person.jpg (image of different person)")
        print("\nThen run this script again.")
        return
    
    # Initialize tester
    tester = DeepFaceModelTester(test_dir)
    
    # Run tests
    print("🚀 Starting comprehensive model testing...")
    results = tester.run_comprehensive_test()
    
    if not results:
        print("❌ No test results generated. Please check your test images.")
        return
    
    # Analyze results
    print("\n📊 Analyzing results...")
    analysis = tester.analyze_results(results)
    
    # Print analysis
    tester.print_analysis(analysis)
    
    # Save results
    tester.save_results(results, analysis)
    
    # Generate recommendation
    recommendation = tester.generate_recommendation(analysis)
    print(recommendation)

if __name__ == "__main__":
    main()
