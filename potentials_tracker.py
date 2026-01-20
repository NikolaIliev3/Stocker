"""
Potentials Tracker - Identifies and tracks upcoming investment opportunities
Stores predictions with target price/date and handles verification
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PotentialsTracker:
    """Tracks potential investment opportunities with price/date predictions"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.potentials_file = self.data_dir / "potentials.json"
        self.potentials = []
        self.load()
    
    def load(self):
        """Load potentials from file"""
        if self.potentials_file.exists():
            try:
                with open(self.potentials_file, 'r') as f:
                    data = json.load(f)
                    self.potentials = data.get('potentials', [])
            except Exception as e:
                logger.error(f"Error loading potentials: {e}")
                self.potentials = []
        else:
            self.potentials = []
    
    def save(self):
        """Save potentials to file"""
        try:
            import math
            
            def clean_for_json(obj):
                """Recursively clean data for JSON (convert NaN/Inf to None)"""
                if isinstance(obj, dict):
                    return {k: clean_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_for_json(item) for item in obj]
                elif isinstance(obj, float):
                    if math.isnan(obj) or math.isinf(obj):
                        return None
                    return obj
                else:
                    return obj
            
            data = {
                'potentials': clean_for_json(self.potentials),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.potentials_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving potentials: {e}")
    
    def add_potential(self, symbol: str, current_price: float, target_price: float,
                     target_date: str, confidence: float, reasoning: str,
                     indicators: dict = None) -> Dict:
        """Add a new potential opportunity"""
        # Check for duplicates (same symbol with active status)
        for p in self.potentials:
            if p.get('symbol') == symbol.upper() and p.get('status') == 'active':
                logger.info(f"Potential for {symbol} already exists, skipping")
                return p
        
        potential = {
            "id": len(self.potentials) + 1,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol.upper(),
            "current_price": current_price,
            "target_price": target_price,
            "target_date": target_date,
            "confidence": confidence,
            "reasoning": reasoning,
            "indicators": indicators or {},
            "status": "active",  # active, verified, expired
            "verified": False,
            "was_correct": None,
            "actual_low_price": None,
            "verification_date": None
        }
        
        self.potentials.append(potential)
        self.save()
        logger.info(f"Added potential: {symbol} @ ${target_price:.2f} by {target_date}")
        return potential
    
    def get_active_potentials(self) -> List[Dict]:
        """Get all active potentials"""
        return [p for p in self.potentials if p.get('status') == 'active']
    
    def get_potential_by_id(self, potential_id: int) -> Optional[Dict]:
        """Get a potential by ID"""
        for p in self.potentials:
            if p.get('id') == potential_id:
                return p
        return None
    
    def verify_potential(self, potential_id: int, was_correct: bool, 
                        actual_low_price: float = None) -> Optional[Dict]:
        """Verify a potential with outcome"""
        potential = self.get_potential_by_id(potential_id)
        if not potential or potential.get('status') != 'active':
            return None
        
        potential['status'] = 'verified'
        potential['verified'] = True
        potential['was_correct'] = was_correct
        potential['verification_date'] = datetime.now().isoformat()
        
        if actual_low_price is not None:
            potential['actual_low_price'] = actual_low_price
        
        self.save()
        return potential
    
    def check_for_verification(self, data_fetcher, calibration_manager=None) -> List[Dict]:
        """Check active potentials for verification conditions and feed to Megamind"""
        verified_items = []
        active = self.get_active_potentials()
        
        for p in active:
            try:
                symbol = p.get('symbol')
                target_price = p.get('target_price', 0)
                original_price = p.get('current_price', 0)
                target_date_str = p.get('target_date', '')
                
                # Get current price
                current_data = data_fetcher.fetch_stock_data(symbol)
                current_price = current_data.get('price', 0)
                
                if current_price <= 0:
                    continue
                
                # Check 1: Has price hit target?
                price_hit = current_price <= target_price
                
                # Check 2: Has target date passed?
                try:
                    target_date = datetime.fromisoformat(target_date_str)
                    date_passed = datetime.now() > target_date
                except:
                    date_passed = False
                
                # Determine outcome
                if price_hit:
                    # Price reached target before date - SUCCESS
                    self.verify_potential(p['id'], True, current_price)
                    p['was_correct'] = True
                    verified_items.append(p)
                    logger.info(f"Potential {symbol} verified CORRECT - price hit ${current_price:.2f}")
                    
                    # Feed to Megamind for recalibration
                    if calibration_manager:
                        try:
                            calibration_manager.record_result(
                                strategy='trading',
                                symbol=symbol,
                                action='BUY',  # Potentials are buy opportunities
                                entry_price=original_price,
                                target_price=target_price,
                                actual_price=current_price,
                                was_correct=True
                            )
                            logger.info(f"🧠 Megamind learning from {symbol} potential (CORRECT)")
                        except Exception as e:
                            logger.warning(f"Failed to feed Megamind for {symbol}: {e}")
                            
                elif date_passed:
                    # Date passed but price never hit target - FAIL
                    self.verify_potential(p['id'], False, current_price)
                    p['was_correct'] = False
                    verified_items.append(p)
                    logger.info(f"Potential {symbol} verified INCORRECT - date passed, price at ${current_price:.2f}")
                    
                    # Feed to Megamind for recalibration
                    if calibration_manager:
                        try:
                            calibration_manager.record_result(
                                strategy='trading',
                                symbol=symbol,
                                action='BUY',
                                entry_price=original_price,
                                target_price=target_price,
                                actual_price=current_price,
                                was_correct=False
                            )
                            logger.info(f"🧠 Megamind learning from {symbol} potential (INCORRECT)")
                        except Exception as e:
                            logger.warning(f"Failed to feed Megamind for {symbol}: {e}")
                    
            except Exception as e:
                logger.error(f"Error checking potential {p.get('id')}: {e}")
        
        return verified_items
    
    def delete_potential(self, potential_id: int) -> bool:
        """Delete a potential"""
        potential = self.get_potential_by_id(potential_id)
        if potential:
            self.potentials.remove(potential)
            self.save()
            return True
        return False
    
    def get_statistics(self) -> Dict:
        """Get statistics about potentials"""
        total = len(self.potentials)
        active = len([p for p in self.potentials if p.get('status') == 'active'])
        verified = len([p for p in self.potentials if p.get('verified', False)])
        correct = len([p for p in self.potentials if p.get('was_correct') is True])
        
        accuracy = (correct / verified * 100) if verified > 0 else 0
        
        return {
            'total': total,
            'active': active,
            'verified': verified,
            'correct': correct,
            'incorrect': verified - correct,
            'accuracy': accuracy
        }
