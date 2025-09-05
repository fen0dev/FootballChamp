import numpy as np
from sklearn.metrics import log_loss, accuracy_score, precision_recall_fscore_support
from typing import Dict, Optional

def brier_score(y_true_onehot: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score per probabilità multiclass"""
    return np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1))

def multi_log_loss(y_true_labels, y_prob) -> float:
    """Log loss per classificazione multiclass"""
    return log_loss(y_true_labels, y_prob, labels=[0, 1, 2])

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    🚀 Expected Calibration Error (ECE) - misura quanto sono calibrate le probabilità
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        for class_idx in range(3):  # 3 classi: home, draw, away
            # Trova predizioni in questo bin per questa classe
            in_bin = (y_prob[:, class_idx] > bin_lower) & (y_prob[:, class_idx] <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Accuratezza in questo bin
                accuracy_in_bin = (y_true[in_bin] == class_idx).mean()
                # Confidence media in questo bin
                avg_confidence_in_bin = y_prob[in_bin, class_idx].mean()
                # ECE contribution
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)

def betting_metrics(y_true: np.ndarray, y_prob: np.ndarray, 
                   market_probs: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    🚀 Metriche per valutazione betting
    """
    metrics = {}
    
    # Predizioni discrete
    y_pred = np.argmax(y_prob, axis=1)
    
    # Accuracy base
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    
    # Precision/Recall per classe
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    outcome_names = ['home_win', 'draw', 'away_win']
    for i, outcome in enumerate(outcome_names):
        if i < len(precision):
            metrics[f'precision_{outcome}'] = float(precision[i])
            metrics[f'recall_{outcome}'] = float(recall[i])
            metrics[f'f1_{outcome}'] = float(f1[i])
            metrics[f'support_{outcome}'] = int(support[i]) if i < len(support) else 0
    
    # Confidence metrics
    max_probs = np.max(y_prob, axis=1)
    metrics['avg_confidence'] = float(np.mean(max_probs))
    metrics['high_confidence_rate'] = float((max_probs > 0.6).mean())
    
    # 🚀 BETTING SIMULATION (se abbiamo quote market)
    if market_probs is not None:
        model_confidence = np.max(y_prob, axis=1)
        market_confidence = np.max(market_probs, axis=1)
        
        # Scommetti quando siamo più confident del mercato
        confident_bets = model_confidence > market_confidence
        
        if confident_bets.sum() > 0:
            correct_confident_bets = (y_pred[confident_bets] == y_true[confident_bets]).sum()
            metrics['confident_bets_count'] = int(confident_bets.sum())
            metrics['confident_bets_accuracy'] = float(correct_confident_bets / confident_bets.sum())
            
            # ROI approssimativo (semplificato)
            avg_market_prob = np.mean(market_probs[confident_bets], axis=0)
            avg_odds = 1 / (avg_market_prob + 1e-10)  # Evita divisione per zero
            estimated_roi = (correct_confident_bets * np.mean(avg_odds) - confident_bets.sum()) / confident_bets.sum()
            metrics['estimated_roi'] = float(estimated_roi)
        else:
            metrics.update({
                'confident_bets_count': 0,
                'confident_bets_accuracy': 0.0,
                'estimated_roi': 0.0
            })
    
    return metrics

def comprehensive_evaluation(y_true: np.ndarray, y_prob: np.ndarray,
                           market_probs: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    🚀 Valutazione comprehensive con tutte le metriche
    """
    # One-hot encoding per Brier score
    y_onehot = np.eye(3)[y_true]
    
    # Metriche base
    metrics = {
        'log_loss': multi_log_loss(y_true, y_prob),
        'brier_score': brier_score(y_onehot, y_prob),
        'expected_calibration_error': expected_calibration_error(y_true, y_prob)
    }
    
    # Metriche betting
    betting_metrics_dict = betting_metrics(y_true, y_prob, market_probs)
    metrics.update(betting_metrics_dict)
    
    # 🚀 PERFORMANCE GRADE
    log_loss_val = metrics['log_loss']
    accuracy_val = metrics['accuracy']
    
    if log_loss_val < 1.0 and accuracy_val > 0.55:
        grade = "Excellent"
    elif log_loss_val < 1.05 and accuracy_val > 0.50:
        grade = "Good"
    elif log_loss_val < 1.1 and accuracy_val > 0.45:
        grade = "Fair"
    else:
        grade = "Poor"
    
    metrics['performance_grade'] = grade
    
    return metrics

def rank_correlation(predictions1: np.ndarray, predictions2: np.ndarray) -> float:
    """
    🚀 Spearman correlation tra due set di predizioni
    Utile per confrontare modelli
    """
    from scipy.stats import spearmanr
    
    # Usa la probabilità massima come ranking
    rank1 = np.max(predictions1, axis=1)
    rank2 = np.max(predictions2, axis=1)
    
    correlation, _ = spearmanr(rank1, rank2)
    return float(correlation) if not np.isnan(correlation) else 0.0

def prediction_entropy(y_prob: np.ndarray) -> np.ndarray:
    """
    🚀 Calcola entropia delle predizioni (misura di incertezza)
    """
    # Evita log(0)
    y_prob_safe = np.clip(y_prob, 1e-10, 1.0)
    entropy = -np.sum(y_prob_safe * np.log(y_prob_safe), axis=1)
    return entropy

def market_beat_rate(y_true: np.ndarray, model_probs: np.ndarray, 
                    market_probs: np.ndarray) -> Dict[str, float]:
    """
    🚀 Calcola quanto spesso il modello batte il mercato
    """
    model_pred = np.argmax(model_probs, axis=1)
    market_pred = np.argmax(market_probs, axis=1)
    
    model_correct = (model_pred == y_true)
    market_correct = (market_pred == y_true)
    
    # Casi dove il modello è giusto e il mercato sbagliato
    model_beats_market = model_correct & ~market_correct
    # Casi dove il mercato è giusto e il modello sbagliato
    market_beats_model = market_correct & ~model_correct
    # Casi dove entrambi sono giusti o entrambi sbagliano
    tie_cases = (model_correct == market_correct)
    
    total = len(y_true)
    
    return {
        'model_beats_market_rate': float(model_beats_market.sum() / total),
        'market_beats_model_rate': float(market_beats_model.sum() / total),
        'tie_rate': float(tie_cases.sum() / total),
        'model_advantage': float((model_beats_market.sum() - market_beats_model.sum()) / total)
    }