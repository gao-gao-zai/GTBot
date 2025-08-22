import time
import asyncio
from functools import wraps
from collections import defaultdict
from datetime import datetime, timedelta
import statistics
import threading
from typing import Dict, List, Any
import json

# 存储统计数据的全局字典
_func_stats = defaultdict(lambda: {
    'total_time': 0.0,
    'count': 0,
    'min_time': float('inf'),
    'max_time': 0.0,
    'execution_times': [],  # 存储所有执行时间用于计算统计指标
    'first_call': None,     # 首次调用时间
    'last_call': None,      # 最后调用时间
    'error_count': 0,       # 错误次数
    'success_count': 0,     # 成功次数
})

_stats_lock = asyncio.Lock()    # 异步锁
_sync_stats_lock = threading.Lock()  # 同步锁

def async_timer(func):
    """异步函数计时装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        current_time = datetime.now()
        error_occurred = False
        
        try:
            result = await func(*args, **kwargs)
        except Exception as e:
            error_occurred = True
            raise
        finally:
            elapsed = (time.perf_counter() - start_time) * 1000
            
            async with _stats_lock:
                stats = _func_stats[func.__name__]
                stats['total_time'] += elapsed
                stats['count'] += 1
                stats['min_time'] = min(stats['min_time'], elapsed)
                stats['max_time'] = max(stats['max_time'], elapsed)
                stats['execution_times'].append(elapsed)
                
                if stats['first_call'] is None:
                    stats['first_call'] = current_time
                stats['last_call'] = current_time
                
                if error_occurred:
                    stats['error_count'] += 1
                else:
                    stats['success_count'] += 1
        
        return result
    return wrapper

def sync_timer(func):
    """同步函数计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        current_time = datetime.now()
        error_occurred = False
        
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            error_occurred = True
            raise
        finally:
            elapsed = (time.perf_counter() - start_time) * 1000
            
            with _sync_stats_lock:
                stats = _func_stats[func.__name__]
                stats['total_time'] += elapsed
                stats['count'] += 1
                stats['min_time'] = min(stats['min_time'], elapsed)
                stats['max_time'] = max(stats['max_time'], elapsed)
                stats['execution_times'].append(elapsed)
                
                if stats['first_call'] is None:
                    stats['first_call'] = current_time
                stats['last_call'] = current_time
                
                if error_occurred:
                    stats['error_count'] += 1
                else:
                    stats['success_count'] += 1
        
        return result
    return wrapper

def _calculate_percentiles(times: List[float]) -> Dict[str, float]:
    """计算百分位数"""
    if not times:
        return {}
    
    sorted_times = sorted(times)
    return {
        'P50': statistics.median(sorted_times),
        'P90': sorted_times[int(len(sorted_times) * 0.9)],
        'P95': sorted_times[int(len(sorted_times) * 0.95)],
        'P99': sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) >= 100 else sorted_times[-1]
    }

def _format_time(ms: float) -> str:
    """格式化时间显示"""
    if ms < 1:
        return f"{ms:.3f}ms"
    elif ms < 1000:
        return f"{ms:.2f}ms"
    else:
        return f"{ms/1000:.2f}s"

def _format_duration(td: timedelta) -> str:
    """格式化时间间隔"""
    total_seconds = int(td.total_seconds())
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    if days > 0:
        return f"{days}天{hours}小时{minutes}分{seconds}秒"
    elif hours > 0:
        return f"{hours}小时{minutes}分{seconds}秒"
    elif minutes > 0:
        return f"{minutes}分{seconds}秒"
    else:
        return f"{seconds}秒"

def print_stats(sort_by='total_time', top_n=None, show_details=True):
    """
    打印函数执行统计信息
    
    Args:
        sort_by: 排序方式 ('total_time', 'count', 'avg_time', 'max_time')
        top_n: 只显示前N个函数，None表示显示全部
        show_details: 是否显示详细统计信息
    """
    if not _func_stats:
        print("📊 暂无函数统计数据")
        return
    
    print("\n" + "="*120)
    print("🚀 函数性能统计报告")
    print("="*120)
    
    # 排序选项映射
    sort_options = {
        'total_time': lambda x: x[1]['total_time'],
        'count': lambda x: x[1]['count'],
        'avg_time': lambda x: x[1]['total_time'] / x[1]['count'] if x[1]['count'] > 0 else 0,
        'max_time': lambda x: x[1]['max_time'],
        'error_rate': lambda x: x[1]['error_count'] / x[1]['count'] if x[1]['count'] > 0 else 0
    }
    
    # 按指定方式排序
    sorted_stats = sorted(
        _func_stats.items(), 
        key=sort_options.get(sort_by, sort_options['total_time']), 
        reverse=True
    )
    
    if top_n:
        sorted_stats = sorted_stats[:top_n]
    
    # 基础统计表格
    print(f"{'函数名':<25} | {'调用次数':>8} | {'总耗时':>12} | {'平均耗时':>12} | {'最小耗时':>12} | {'最大耗时':>12} | {'成功率':>8}")
    print("-" * 120)
    
    for name, data in sorted_stats:
        if data['count'] > 0:
            avg_time = data['total_time'] / data['count']
            success_rate = (data['success_count'] / data['count']) * 100 if data['count'] > 0 else 0
            
            print(f"{name[:24]:<25} | {data['count']:>8} | "
                  f"{_format_time(data['total_time']):>12} | "
                  f"{_format_time(avg_time):>12} | "
                  f"{_format_time(data['min_time']):>12} | "
                  f"{_format_time(data['max_time']):>12} | "
                  f"{success_rate:>7.1f}%")
    
    if show_details:
        print("\n" + "="*120)
        print("📈 详细统计信息")
        print("="*120)
        
        for name, data in sorted_stats:
            if data['count'] == 0:
                continue
                
            print(f"\n🔍 函数: {name}")
            print("-" * 80)
            
            # 基础信息
            avg_time = data['total_time'] / data['count']
            success_rate = (data['success_count'] / data['count']) * 100
            error_rate = (data['error_count'] / data['count']) * 100
            
            print(f"  📊 基础统计:")
            print(f"    总调用次数: {data['count']}")
            print(f"    成功次数: {data['success_count']}")
            print(f"    失败次数: {data['error_count']}")
            print(f"    成功率: {success_rate:.1f}%")
            print(f"    错误率: {error_rate:.1f}%")
            
            print(f"  ⏱️  执行时间:")
            print(f"    总耗时: {_format_time(data['total_time'])}")
            print(f"    平均耗时: {_format_time(avg_time)}")
            print(f"    最小耗时: {_format_time(data['min_time'])}")
            print(f"    最大耗时: {_format_time(data['max_time'])}")
            
            # 计算标准差和百分位数
            if len(data['execution_times']) > 1:
                std_dev = statistics.stdev(data['execution_times'])
                percentiles = _calculate_percentiles(data['execution_times'])
                
                print(f"    标准差: {_format_time(std_dev)}")
                print(f"    中位数 (P50): {_format_time(percentiles.get('P50', 0))}")
                print(f"    P90: {_format_time(percentiles.get('P90', 0))}")
                print(f"    P95: {_format_time(percentiles.get('P95', 0))}")
                print(f"    P99: {_format_time(percentiles.get('P99', 0))}")
            
            # 时间信息
            if data['first_call'] and data['last_call']:
                duration = data['last_call'] - data['first_call']
                print(f"  📅 时间信息:")
                print(f"    首次调用: {data['first_call'].strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"    最后调用: {data['last_call'].strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"    监控时长: {_format_duration(duration)}")
                
                if duration.total_seconds() > 0:
                    calls_per_second = data['count'] / duration.total_seconds()
                    print(f"    调用频率: {calls_per_second:.2f}次/秒")
            
            # 性能评估
            print(f"  🎯 性能评估:")
            if avg_time < 1:
                performance_level = "🟢 极快"
            elif avg_time < 10:
                performance_level = "🟢 很快"
            elif avg_time < 100:
                performance_level = "🟡 一般"
            elif avg_time < 1000:
                performance_level = "🟠 较慢"
            else:
                performance_level = "🔴 很慢"
            
            print(f"    性能等级: {performance_level}")
            
            if error_rate > 0:
                if error_rate < 1:
                    reliability_level = "🟢 非常可靠"
                elif error_rate < 5:
                    reliability_level = "🟡 比较可靠"
                elif error_rate < 10:
                    reliability_level = "🟠 一般可靠"
                else:
                    reliability_level = "🔴 不够可靠"
                print(f"    可靠性等级: {reliability_level}")
    
    print("\n" + "="*120)

def get_stats_summary() -> Dict[str, Any]:
    """获取统计数据摘要"""
    total_functions = len(_func_stats)
    total_calls = sum(data['count'] for data in _func_stats.values())
    total_time = sum(data['total_time'] for data in _func_stats.values())
    total_errors = sum(data['error_count'] for data in _func_stats.values())
    
    return {
        'total_functions': total_functions,
        'total_calls': total_calls,
        'total_time_ms': total_time,
        'total_errors': total_errors,
        'overall_error_rate': (total_errors / total_calls * 100) if total_calls > 0 else 0,
        'average_call_time': total_time / total_calls if total_calls > 0 else 0
    }

def export_stats_to_json(filename: str = None) -> str:
    """导出统计数据到JSON文件"""
    if filename is None:
        filename = f"performance_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # 转换数据格式（移除不能序列化的对象）
    export_data = {}
    for name, data in _func_stats.items():
        export_data[name] = {
            'total_time': data['total_time'],
            'count': data['count'],
            'min_time': data['min_time'] if data['min_time'] != float('inf') else 0,
            'max_time': data['max_time'],
            'execution_times': data['execution_times'],
            'first_call': data['first_call'].isoformat() if data['first_call'] else None,
            'last_call': data['last_call'].isoformat() if data['last_call'] else None,
            'error_count': data['error_count'],
            'success_count': data['success_count'],
        }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    print(f"📄 统计数据已导出到: {filename}")
    return filename

def clear_stats():
    """清除所有统计数据"""
    global _func_stats
    _func_stats.clear()
    print("🧹 统计数据已清除")

def reset_function_stats(func_name: str):
    """重置指定函数的统计数据"""
    if func_name in _func_stats:
        del _func_stats[func_name]
        print(f"🔄 函数 '{func_name}' 的统计数据已重置")
    else:
        print(f"⚠️  未找到函数 '{func_name}' 的统计数据")

# 使用示例
if __name__ == "__main__":
    # 示例异步函数
    @async_timer
    async def async_example():
        await asyncio.sleep(0.01)
        return "异步任务完成"
    
    # 示例同步函数
    @sync_timer
    def sync_example(n):
        time.sleep(0.005)
        if n % 10 == 0:
            raise ValueError("模拟错误")
        return n * 2
    
    # 运行示例
    async def run_example():
        # 测试异步函数
        for _ in range(5):
            await async_example()
        
        # 测试同步函数
        for i in range(15):
            try:
                sync_example(i)
            except ValueError:
                pass  # 忽略模拟的错误
        
        # 显示统计信息
        print_stats(show_details=True)
        
        # 显示摘要
        summary = get_stats_summary()
        print(f"\n📋 总体摘要:")
        print(f"  监控函数数: {summary['total_functions']}")
        print(f"  总调用次数: {summary['total_calls']}")
        print(f"  总耗时: {_format_time(summary['total_time_ms'])}")
        print(f"  总错误次数: {summary['total_errors']}")
        print(f"  总体错误率: {summary['overall_error_rate']:.1f}%")
        print(f"  平均调用时间: {_format_time(summary['average_call_time'])}")
    
    # 运行示例（如果直接执行此文件）
    # asyncio.run(run_example())