import random
import threading

from ... import ax


@ax.task
def simple_return_task():
    return 1

def simple_return():
    t = simple_return_task().wait()
    if not t.succeeded or t.result != 1:
        return False
    return True

@ax.task
def branch_true_false_task(b):
    yield ax.sleep(0)

    if b:
        yield ax.success(1)
    else:
        yield ax.cancel()

def branch_true_1():
    t = branch_true_false_task(True).wait()
    return t.succeeded and t.result == 1

def branch_false_cancel():
    t = branch_true_false_task(False).wait()
    return not t.succeeded

@ax.task
def sleep_1_task():
    yield ax.sleep(1.0)
    return 1

def sleep_1():
    t = sleep_1_task().wait()
    return t.succeeded and t.result == 1

@ax.task
def propagate_task_1():
    yield ax.sleep(0)
    return 1

@ax.task
def propagate_task():
    yield ax.propagate( propagate_task_1() )

def propagate():
    t = propagate_task().wait()
    return t.succeeded and t.result == 1


@ax.task
def wait_multi_task_1():
    yield ax.sleep( random.uniform(0.0, 1.0) )

@ax.task
def wait_multi_task():
    yield ax.wait([wait_multi_task_1() for _ in range(8)])
    return 1

def wait_multi():
    t = sleep_1_task().wait()
    return t.succeeded and t.result == 1


@ax.task
def compute_in_single_thread_task_0(count):
    result = 0
    for i in range(count):
        result += i
        yield ax.sleep(0)
    return result

@ax.task
def compute_in_single_thread_task():
    yield ax.wait(tasks := [ compute_in_single_thread_task_0(i) for i in range(128) ])
    return sum([ task.result for task in tasks ])

def compute_in_single_thread():
    t = compute_in_single_thread_task().wait()
    return t.succeeded and t.result == 341376

@ax.task
def thread_task():
    prev_t, new_t = ax.get_current_thread(), ax.Thread(name='temp')
    yield ax.switch_to(new_t)

    yield ax.sleep( random.uniform(0.0, 1.0) )

    yield ax.switch_to(prev_t)

    new_t.dispose()

    return 1

def thread():
    t = thread_task().wait()
    return t.succeeded and t.result == 1

@ax.task
def multi_thread_task(main_call=True, data = None):
    if main_call:
        data = []
        yield ax.wait(tasks := [ multi_thread_task(main_call=False, data=data) for _ in range(8)])
        return sum(data)
    else:
        prev_t, new_t = ax.get_current_thread(), ax.Thread(name='temp')
        yield ax.switch_to(new_t)
        yield ax.sleep( random.uniform(0.0, 1.0) )
        data.append(1)
        yield ax.switch_to(prev_t)
        new_t.dispose()

def multi_thread():
    t = multi_thread_task().wait()
    return t.succeeded and t.result == 8

@ax.task
def future_group_task(fg : ax.FutureGroup):
    yield ax.attach_to(fg)

    yield ax.sleep(999.0)

def future_group():
    fg = ax.FutureGroup('fg')

    t = future_group_task(fg)
    if fg.count != 1:
        return False

    t.cancel()

    if fg.count != 0:
        return False

    t = future_group_task(fg)

    fg.dispose()

    if t.succeeded:
        return False
    if fg.count != 0:
        return False

    t = future_group_task(fg)
    if t.succeeded:
        return False

    return True


@ax.task
def detach_task_child(thread : ax.Thread):
    yield ax.switch_to(thread)

    yield ax.detach()
    yield ax.sleep(1.0)
    return True

@ax.task
def detach_task_tr(fg : ax.FutureGroup, thread : ax.Thread):
    yield ax.switch_to(thread)
    yield ax.attach_to(fg)

    yield ax.detach()
    yield ax.sleep(1.0)
    return True

@ax.task
def detach_task():
    thread = ax.Thread()
    fg = ax.FutureGroup('fg')

    child_task = detach_task_child(thread)
    tr_task = detach_task_tr(fg, thread)

    yield ax.sleep(0.5)

    fg.cancel_all()

    yield ax.wait([child_task,tr_task])

    thread.dispose()
    fg.dispose()

    if not child_task.succeeded:
        return False

    if not tr_task.succeeded:
        return False

    return True

def detach():
    return detach_task().wait().result

@ax.task
def finish_exception_task(ev):
    try:
        yield ax.sleep(1.0)
        yield ax.success()
    except ax.TaskFinishException as e:
        ...
    ev.set()

def finish_exception():
    ev = threading.Event()
    finish_exception_task(ev).wait()
    if not ev.is_set():
        return False

    ev = threading.Event()
    t = finish_exception_task(ev)
    t.cancel()
    if not ev.is_set():
        return False
    return True

@ax.task
def child_tasks_task_2():
    yield ax.sleep(999)

@ax.task
def child_tasks_task_1(task_ar):
    task_ar.append( child_tasks_task_2())
    yield ax.sleep(999)

@ax.task
def child_tasks_task_0(task_ar):
    task_ar.append( child_tasks_task_1(task_ar))
    yield ax.sleep(999)

@ax.task
def child_tasks_task():
    task_ar = []
    t =  child_tasks_task_0(task_ar)
    task_ar.append(t)
    yield ax.sleep(0.5)
    t.cancel()
    return all(task.finished for task in task_ar)

def child_tasks():
    return child_tasks_task().wait().result

@ax.task
def thread_pool_task_0( pool : ax.ThreadPool, i ):
    yield ax.switch_to(pool)
    return True

@ax.task
def thread_pool_task():
    pool = ax.ThreadPool(name='Pool')
    yield ax.wait([ thread_pool_task_0(pool, i) for i in range(pool.count) ])
    pool.dispose()
    return True

def thread_pool():
    return thread_pool_task().wait().result

@ax.task
def futureset_task_0():
    yield ax.sleep(1)
    return True

@ax.task
def futureset_task():

    fs = ax.FutureSet()
    fs.add( futureset_task_0() )
    fs.add( futureset_task_0() )

    if fs.count != 2:
        return False

    fs.cancel_all()

    if fs.count != 0:
        return False

    fs.add( futureset_task_0() )
    fs.add( futureset_task_0() )

    if len(fs.fetch(finished=True)) != 0:
        return False
    if len(fs.fetch(succeeded=True)) != 0:
        return False

    while fs.count != 0:
        for _ in fs.fetch(finished=True):
            ...
        yield ax.sleep(0)

    return True

def futureset():
    return futureset_task().wait().result

@ax.task
def call_soon_task():
    ar = [False]
    
    thread = ax.get_current_thread()
    thread.call_soon(lambda: ar.append(True))
    yield ax.sleep(0.5)
    
    if ar[-1] != True:
        return False

    d = thread.call_soon(lambda: ar.append(False))
    d.dispose()
    yield ax.sleep(0.5)    
    
    if ar[-1] != True:
        return False

    return True

def call_soon():
    return call_soon_task().wait().result


@ax.task
def future_generator_task_0(i) -> int:
    yield ax.sleep(0.1)
    return 1
    
@ax.task
def future_generator_task():
    r = 0
    for value in ax.FutureGenerator( ( (future_generator_task_0(i),None) for i in range(10)), max_parallel=2 ):
        if value is None:
            yield ax.sleep(0)
        else:
            fut, param = value
            r += fut.result
            
    return r == 10
    
def future_generator():
    return future_generator_task().wait().result

def run_test():
    """
    """
    log_level = ax.g_log.get_level()
    ax.g_log.set_level(2)

    ax.clear()
    tests = [simple_return, branch_true_1, branch_false_cancel,
             sleep_1, propagate, wait_multi, future_group, child_tasks, detach, futureset,
             compute_in_single_thread, thread, multi_thread,
             finish_exception, thread_pool, call_soon, future_generator,  
             ]

    tests_result = []

    for func in tests:
        print(f'Testing {func.__name__}...')
        result = func()
        tests_result.append( result )
        print(f"{func.__name__} {'OK' if result else 'FAIL'}" )
        if not result:
            break

    ax.g_debug.print_info()

    ax.clear()

    ax.g_log.set_level(log_level)



