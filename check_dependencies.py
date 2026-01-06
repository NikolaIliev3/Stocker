"""
Dependency Security Checker
Check dependencies for known vulnerabilities
Run: pip install safety && python check_dependencies.py
"""
import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check for vulnerable dependencies"""
    try:
        logger.info("Checking dependencies for known vulnerabilities...")
        result = subprocess.run(
            ['safety', 'check', '--json'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            logger.error("⚠️ Security vulnerabilities found in dependencies!")
            print(result.stdout)
            
            # Try to parse JSON output
            try:
                import json
                vulnerabilities = json.loads(result.stdout)
                if vulnerabilities:
                    logger.error(f"Found {len(vulnerabilities)} vulnerabilities:")
                    for vuln in vulnerabilities:
                        logger.error(f"  - {vuln.get('package', 'unknown')}: {vuln.get('vulnerability', 'unknown')}")
            except:
                pass
            
            return False
        else:
            logger.info("✅ No known vulnerabilities found in dependencies")
            return True
            
    except FileNotFoundError:
        logger.warning("⚠️ 'safety' not installed. Install with: pip install safety")
        logger.info("To check dependencies, run: pip install safety && safety check")
        return None
    except subprocess.TimeoutExpired:
        logger.error("⚠️ Dependency check timed out")
        return False
    except Exception as e:
        logger.error(f"⚠️ Error checking dependencies: {e}")
        return False


if __name__ == "__main__":
    success = check_dependencies()
    sys.exit(0 if success else 1)
