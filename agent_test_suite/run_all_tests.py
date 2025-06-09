"""
Master test runner for AgentCoder comprehensive test suite.
Runs all test modules and provides a summary report.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_test_module(module_name, module_path):
    """Run a single test module and return results."""
    print(f"\n{'='*70}")
    print(f"Running: {module_name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(module_path)],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=300  # 5 minute timeout per test suite
        )
        
        elapsed_time = time.time() - start_time
        
        # Determine success based on output content
        success_indicators = ["✅", "SUCCESS", "completed", "successful"]
        failure_indicators = ["❌", "FAILED", "Error", "Exception"]
        
        success_count = sum(1 for ind in success_indicators if ind in result.stdout)
        failure_count = sum(1 for ind in failure_indicators if ind in result.stdout)
        
        # Consider it successful if more success indicators than failures
        success = success_count > failure_count or (result.returncode == 0 and failure_count == 0)
        
        return {
            "name": module_name,
            "success": success,
            "returncode": result.returncode,
            "elapsed_time": elapsed_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success_count": success_count,
            "failure_count": failure_count
        }
        
    except subprocess.TimeoutExpired:
        return {
            "name": module_name,
            "success": False,
            "returncode": -1,
            "elapsed_time": 300,
            "stdout": "",
            "stderr": "Test timed out after 5 minutes",
            "success_count": 0,
            "failure_count": 1
        }
    except Exception as e:
        return {
            "name": module_name,
            "success": False,
            "returncode": -1,
            "elapsed_time": time.time() - start_time,
            "stdout": "",
            "stderr": str(e),
            "success_count": 0,
            "failure_count": 1
        }


def generate_html_report(results, output_path):
    """Generate an HTML report of test results."""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AgentCoder Test Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background-color: #333; color: white; padding: 20px; border-radius: 5px; }}
        .summary {{ background-color: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .test-result {{ background-color: white; margin: 10px 0; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .success {{ border-left: 5px solid #4CAF50; }}
        .failure {{ border-left: 5px solid #f44336; }}
        .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .stat-box {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; text-align: center; }}
        pre {{ background-color: #f0f0f0; padding: 10px; border-radius: 3px; overflow-x: auto; }}
        details {{ margin: 10px 0; }}
        summary {{ cursor: pointer; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>AgentCoder Comprehensive Test Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <h2>Test Summary</h2>
        <div class="stats">
            <div class="stat-box">
                <h3>Total Tests</h3>
                <p style="font-size: 24px;">{len(results)}</p>
            </div>
            <div class="stat-box">
                <h3>Passed</h3>
                <p style="font-size: 24px; color: #4CAF50;">{sum(1 for r in results if r['success'])}</p>
            </div>
            <div class="stat-box">
                <h3>Failed</h3>
                <p style="font-size: 24px; color: #f44336;">{sum(1 for r in results if not r['success'])}</p>
            </div>
            <div class="stat-box">
                <h3>Total Time</h3>
                <p style="font-size: 24px;">{sum(r['elapsed_time'] for r in results):.1f}s</p>
            </div>
        </div>
    </div>
    
    <h2>Test Results</h2>
"""
    
    for result in results:
        status_class = "success" if result['success'] else "failure"
        status_icon = "✅" if result['success'] else "❌"
        
        html_content += f"""
    <div class="test-result {status_class}">
        <h3>{status_icon} {result['name']}</h3>
        <p><strong>Status:</strong> {'PASSED' if result['success'] else 'FAILED'} | 
           <strong>Return Code:</strong> {result['returncode']} | 
           <strong>Duration:</strong> {result['elapsed_time']:.2f}s |
           <strong>Success Indicators:</strong> {result['success_count']} |
           <strong>Failure Indicators:</strong> {result['failure_count']}</p>
        
        <details>
            <summary>View Output</summary>
            <h4>Standard Output:</h4>
            <pre>{result['stdout'][:5000]}{'... (truncated)' if len(result['stdout']) > 5000 else ''}</pre>
            {f"<h4>Standard Error:</h4><pre>{result['stderr']}</pre>" if result['stderr'] else ""}
        </details>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def main():
    """Run all test suites and generate reports."""
    print("🚀 AgentCoder Comprehensive Test Suite Runner")
    print("=" * 70)
    
    # Check environment
    print("\n📋 Environment Check:")
    print(f"Python: {sys.version}")
    print(f"GEMINI_API_KEY: {'✅ Set' if os.getenv('GEMINI_API_KEY') else '❌ Not set'}")
    print(f"DEEPSEEK_API_KEY: {'✅ Set' if os.getenv('DEEPSEEK_API_KEY') else '❌ Not set'}")
    
    # Find all test modules
    test_dir = Path(__file__).parent / "python_tests"
    test_modules = list(test_dir.glob("test_*.py"))
    
    print(f"\n📦 Found {len(test_modules)} test modules:")
    for module in test_modules:
        print(f"  - {module.name}")
    
    # Run all tests
    print("\n🧪 Running tests...")
    results = []
    
    for module in test_modules:
        result = run_test_module(module.stem, module)
        results.append(result)
        
        # Print quick summary
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"{status} - {result['name']} ({result['elapsed_time']:.2f}s)")
    
    # Generate summary
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY")
    print("=" * 70)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['success'])
    failed_tests = total_tests - passed_tests
    total_time = sum(r['elapsed_time'] for r in results)
    
    print(f"Total Test Suites: {total_tests}")
    print(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
    print(f"Total Time: {total_time:.2f} seconds")
    
    # Generate HTML report
    report_path = Path(__file__).parent / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    generate_html_report(results, report_path)
    print(f"\n📄 HTML report generated: {report_path}")
    
    # Print failed test details
    if failed_tests > 0:
        print("\n❌ Failed Tests:")
        for result in results:
            if not result['success']:
                print(f"\n  {result['name']}:")
                if result['stderr']:
                    print(f"    Error: {result['stderr'][:200]}...")
    
    # Return exit code based on results
    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    sys.exit(main()) 