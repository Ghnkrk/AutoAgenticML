"""
Model Evaluator Module

Pure Python evaluation node that ranks trained models and provides
decision-ready results without training, LLM calls, or data modification.
"""

from typing import Dict, List, Any
import statistics


class ModelEvaluator:
    """
    Deterministic model evaluation and ranking.
    Analyzes trained model metrics and provides structured recommendations.
    """
    
    def __init__(self, primary_metric: str = "accuracy"):
        """
        Initialize ModelEvaluator.
        
        Args:
            primary_metric: Metric to use for ranking (e.g., 'accuracy', 'f1', 'roc_auc')
        """
        self.primary_metric = primary_metric
        
        # Thresholds for confidence levels
        self.HIGH_CONFIDENCE_THRESHOLD = 0.02  # 2% difference
        self.MEDIUM_CONFIDENCE_THRESHOLD = 0.05  # 5% difference
        
    def forward(self, trained_models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate and rank trained models.
        
        Args:
            trained_models: List of dicts with keys: name, path, metrics, timestamp
            
        Returns:
            Dict with model_ranking, best_model, confidence, warnings, recommendation
        """
        if not trained_models:
            return {
                "model_ranking": [],
                "best_model": None,
                "confidence": "low",
                "warnings": ["No trained models to evaluate"],
                "recommendation": "human_review"
            }
        
        # Extract primary metric scores
        model_scores = []
        for model in trained_models:
            metrics = model.get("metrics", {})
            score = metrics.get(self.primary_metric)
            
            if score is None:
                continue
                
            model_scores.append({
                "model_name": model["name"],
                "score": float(score),
                "path": model["path"],
                "all_metrics": metrics
            })
        
        if not model_scores:
            return {
                "model_ranking": [],
                "best_model": None,
                "confidence": "low",
                "warnings": [f"No models have '{self.primary_metric}' metric"],
                "recommendation": "human_review"
            }
        
        # Sort by score (descending - higher is better)
        model_scores.sort(key=lambda x: x["score"], reverse=True)
        
        # Assign ranks
        model_ranking = []
        for rank, model in enumerate(model_scores, 1):
            model_ranking.append({
                "model_name": model["model_name"],
                "score": model["score"],
                "rank": rank,
                "path": model["path"]
            })
        
        # Best model
        best_model = model_ranking[0]
        best_score = best_model["score"]
        
        # Compute score deltas and analyze differences
        score_deltas = []
        for model in model_ranking[1:]:
            delta = best_score - model["score"]
            score_deltas.append(delta)
        
        # Determine confidence based on score spread
        confidence, warnings = self._assess_confidence(
            best_score, score_deltas, model_ranking
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(confidence, warnings)
        
        return {
            "model_ranking": model_ranking,
            "best_model": best_model["model_name"],
            "best_model_path": best_model["path"],
            "best_score": best_score,
            "confidence": confidence,
            "warnings": warnings,
            "recommendation": recommendation,
            "primary_metric": self.primary_metric
        }
    
    def _assess_confidence(
        self,
        best_score: float,
        score_deltas: List[float],
        model_ranking: List[Dict]
    ) -> tuple[str, List[str]]:
        """
        Assess confidence level and generate warnings.
        
        Args:
            best_score: Score of the best model
            score_deltas: Differences between best and other models
            model_ranking: Full ranking list
            
        Returns:
            Tuple of (confidence_level, warnings_list)
        """
        warnings = []
        
        # No other models to compare
        if not score_deltas:
            return "high", warnings
        
        # Check for very close scores (insignificant differences)
        min_delta = min(score_deltas)
        max_delta = max(score_deltas)
        avg_delta = statistics.mean(score_deltas)
        
        # Flag small differences
        if min_delta < self.HIGH_CONFIDENCE_THRESHOLD:
            close_models = [
                model_ranking[i+1]["model_name"]
                for i, delta in enumerate(score_deltas)
                if delta < self.HIGH_CONFIDENCE_THRESHOLD
            ]
            warnings.append(
                f"Models {', '.join(close_models)} are within {self.HIGH_CONFIDENCE_THRESHOLD:.1%} "
                f"of best model (insignificant difference)"
            )
        
        # Check if best model score is low
        if best_score < 0.6:
            warnings.append(
                f"Best model score ({best_score:.4f}) is below 0.6 - consider retraining or feature engineering"
            )
        
        # Check for high variance in scores
        if len(score_deltas) > 1:
            std_dev = statistics.stdev(score_deltas)
            if std_dev > 0.1:
                warnings.append(
                    f"High variance in model performance (std={std_dev:.4f}) - results may be unstable"
                )
        
        # Determine confidence level
        if min_delta >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            confidence = "high"
        elif min_delta >= self.HIGH_CONFIDENCE_THRESHOLD:
            confidence = "medium"
        else:
            confidence = "low"
        
        return confidence, warnings
    
    def _generate_recommendation(
        self,
        confidence: str,
        warnings: List[str]
    ) -> str:
        """
        Generate recommendation based on confidence and warnings.
        
        Args:
            confidence: Confidence level (high/medium/low)
            warnings: List of warning messages
            
        Returns:
            Recommendation string
        """
        # If there are critical warnings or low confidence, recommend human review
        critical_warnings = [
            w for w in warnings
            if "below 0.6" in w or "unstable" in w
        ]
        
        if critical_warnings or confidence == "low":
            return "human_review"
        
        # Otherwise, models can be accepted
        return "accept_models"
