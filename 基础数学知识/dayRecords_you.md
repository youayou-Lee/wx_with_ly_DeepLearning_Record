## 2025-2.26
### 随机梯度下降，梯度下降，小批量梯度下降

今天根据李沐的《动手写深度学习》中 MBGD，修改代码，实现BGD发现：

- **学习率的调整** ：对于BGD，损失函数计算的是所有样本的损失，所以loss对w的偏导值会很大，如果沿用MBGD的lr，会发现loss无法收敛，必须降低lr。
- **MBGD的w更新**：对于每一轮epoch，都需要计算一次grad，然后只更新一次，计算grad开销实在太大，而MBGD对小批量的样本计算grad则开销小得多，且每一轮循环可以多次更新w。lr 为 1 算正常，不需要较多轮次便可完成训练


- **矩阵运算 torch.matmul()**：pytorch中的一维数组是以列向量为数学计算约定，而以行向量为表示形式的向量 ,因为此时的size 为 [m] 的tensor 实际上是 列向量


## 2025-2.28
### debug 多线程问题
报错：这有一段很长的报错，请问是为什么：Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 131, in _main
    prepare(preparation_data)
  File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 246, in prepare
    _fixup_main_from_path(data['init_main_from_path'])
  File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 297, in _fixup_main_from_path
    main_content = runpy.run_path(main_path,
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen runpy>", line 291, in run_path
  File "<frozen runpy>", line 98, in _run_module_code
  File "<frozen runpy>", line 88, in _run_code
  File "F:\code\AI\wx_with_ly_DeepLearning_Record\基础数学知识\lr_softmax_speed.py", line 95, in <module>
    train_losses, train_accs, test_accs = train_ch3(net, train_iter, test_iter, loss, num_epochs, optimizer)
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\code\AI\wx_with_ly_DeepLearning_Record\基础数学知识\lr_softmax_speed.py", line 50, in train_ch3
    for X, y in train_iter:
  File "G:\Anaconda\envs\you\Lib\site-packages\torch\utils\data\dataloader.py", line 440, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\site-packages\torch\utils\data\dataloader.py", line 388, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\site-packages\torch\utils\data\dataloader.py", line 1038, in __init__
    w.start()
  File "G:\Anaconda\envs\you\Lib\multiprocessing\process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\multiprocessing\context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\multiprocessing\context.py", line 336, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\multiprocessing\popen_spawn_win32.py", line 46, in __init__
    prep_data = spawn.get_preparation_data(process_obj._name)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 164, in get_preparation_data
    _check_not_importing_main()
  File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 140, in _check_not_importing_main
    raise RuntimeError('''
RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.

        To fix this issue, refer to the "Safe importing of main module"
        section in https://docs.python.org/3/library/multiprocessing.html
        
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 131, in _main
    prepare(preparation_data)
  File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 246, in prepare
    _fixup_main_from_path(data['init_main_from_path'])
  File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 297, in _fixup_main_from_path
    main_content = runpy.run_path(main_path,
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen runpy>", line 291, in run_path
  File "<frozen runpy>", line 98, in _run_module_code
  File "<frozen runpy>", line 88, in _run_code
  File "F:\code\AI\wx_with_ly_DeepLearning_Record\基础数学知识\lr_softmax_speed.py", line 95, in <module>
    train_losses, train_accs, test_accs = train_ch3(net, train_iter, test_iter, loss, num_epochs, optimizer)
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\code\AI\wx_with_ly_DeepLearning_Record\基础数学知识\lr_softmax_speed.py", line 50, in train_ch3
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 122, in spawn_main
    for X, y in train_iter:
  File "G:\Anaconda\envs\you\Lib\site-packages\torch\utils\data\dataloader.py", line 440, in __iter__
    exitcode = _main(fd, parent_sentinel)
               ^^^^^^    return self._get_iterator()
^^^^^^ ^ ^ ^^^^ ^^       ^^^^^^^^^^^^^^
^^  File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 131, in _main
^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\site-packages\torch\utils\data\dataloader.py", line 388, in _get_iterator
    prepare(preparation_data)
  File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 246, in prepare
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^    _fixup_main_from_path(data['init_main_from_path'])
^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 297, in _fixup_main_from_path
  File "G:\Anaconda\envs\you\Lib\site-packages\torch\utils\data\dataloader.py", line 1038, in __init__
    main_content = runpy.run_path(main_path,
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen runpy>", line 291, in run_path
  File "<frozen runpy>", line 98, in _run_module_code
  File "<frozen runpy>", line 88, in _run_code
  File "F:\code\AI\wx_with_ly_DeepLearning_Record\基础数学知识\lr_softmax_speed.py", line 95, in <module>
    w.start()
  File "G:\Anaconda\envs\you\Lib\multiprocessing\process.py", line 121, in start
    train_losses, train_accs, test_accs = train_ch3(net, train_iter, test_iter, loss, num_epochs, optimizer)
                                            self._popen = self._Popen(self) 
 ^^^^^^^^^^^^^^^        ^^  ^^^ ^^^^       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
^^^  File "G:\Anaconda\envs\you\Lib\multiprocessing\context.py", line 224, in _Popen
^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\code\AI\wx_with_ly_DeepLearning_Record\基础数学知识\lr_softmax_speed.py", line 50, in train_ch3
    return _default_context.get_context().Process._Popen(process_obj)
    for X, y in train_iter: 
      File "G:\Anaconda\envs\you\Lib\site-packages\torch\utils\data\dataloader.py", line 440, in __iter__
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\multiprocessing\context.py", line 336, in _Popen
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\site-packages\torch\utils\data\dataloader.py", line 388, in _get_iterator
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\multiprocessing\popen_spawn_win32.py", line 46, in __init__
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^    prep_data = spawn.get_preparation_data(process_obj._name)^
^^^^^^^^^^^^^^^^^^^^^^ ^^  ^ ^^  
       File "G:\Anaconda\envs\you\Lib\site-packages\torch\utils\data\dataloader.py", line 1038, in __init__
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 164, in get_preparation_data
    _check_not_importing_main()
  File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 140, in _check_not_importing_main
    w.start()
  File "G:\Anaconda\envs\you\Lib\multiprocessing\process.py", line 121, in start
    raise RuntimeError('''
RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.

        To fix this issue, refer to the "Safe importing of main module"
        section in https://docs.python.org/3/library/multiprocessing.html
        
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\multiprocessing\context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\multiprocessing\context.py", line 336, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\multiprocessing\popen_spawn_win32.py", line 46, in __init__
    prep_data = spawn.get_preparation_data(process_obj._name)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 164, in get_preparation_data
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 122, in spawn_main
    _check_not_importing_main()
  File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 140, in _check_not_importing_main
    exitcode = _main(fd, parent_sentinel)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 131, in _main
    raise RuntimeError('''
RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.

        To fix this issue, refer to the "Safe importing of main module"
        section in https://docs.python.org/3/library/multiprocessing.html
        
    prepare(preparation_data)
  File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 246, in prepare
    _fixup_main_from_path(data['init_main_from_path'])
  File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 297, in _fixup_main_from_path
    main_content = runpy.run_path(main_path,
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen runpy>", line 291, in run_path
  File "<frozen runpy>", line 98, in _run_module_code
  File "<frozen runpy>", line 88, in _run_code
  File "F:\code\AI\wx_with_ly_DeepLearning_Record\基础数学知识\lr_softmax_speed.py", line 95, in <module>
    train_losses, train_accs, test_accs = train_ch3(net, train_iter, test_iter, loss, num_epochs, optimizer)
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\code\AI\wx_with_ly_DeepLearning_Record\基础数学知识\lr_softmax_speed.py", line 50, in train_ch3
    for X, y in train_iter:
  File "G:\Anaconda\envs\you\Lib\site-packages\torch\utils\data\dataloader.py", line 440, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\site-packages\torch\utils\data\dataloader.py", line 388, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\site-packages\torch\utils\data\dataloader.py", line 1038, in __init__
    w.start()
  File "G:\Anaconda\envs\you\Lib\multiprocessing\process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\multiprocessing\context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\multiprocessing\context.py", line 336, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\multiprocessing\popen_spawn_win32.py", line 46, in __init__
    prep_data = spawn.get_preparation_data(process_obj._name)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 164, in get_preparation_data
    _check_not_importing_main()
  File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 140, in _check_not_importing_main
    raise RuntimeError('''
RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.

        To fix this issue, refer to the "Safe importing of main module"
        section in https://docs.python.org/3/library/multiprocessing.html
        
Traceback (most recent call last):
  File "G:\Anaconda\envs\you\Lib\site-packages\torch\utils\data\dataloader.py", line 1131, in _try_get_data
    data = self._data_queue.get(timeout=timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\queue.py", line 179, in get
    raise Empty
_queue.Empty

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "F:\code\AI\wx_with_ly_DeepLearning_Record\基础数学知识\lr_softmax_speed.py", line 95, in <module>
    train_losses, train_accs, test_accs = train_ch3(net, train_iter, test_iter, loss, num_epochs, optimizer)
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\code\AI\wx_with_ly_DeepLearning_Record\基础数学知识\lr_softmax_speed.py", line 50, in train_ch3
    for X, y in train_iter:
  File "G:\Anaconda\envs\you\Lib\site-packages\torch\utils\data\dataloader.py", line 630, in __next__
    data = self._next_data()
           ^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\site-packages\torch\utils\data\dataloader.py", line 1327, in _next_data
    idx, data = self._get_data()
                ^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\site-packages\torch\utils\data\dataloader.py", line 1283, in _get_data
    success, data = self._try_get_data()
                    ^^^^^^^^^^^^^^^^^^^^
  File "G:\Anaconda\envs\you\Lib\site-packages\torch\utils\data\dataloader.py", line 1144, in _try_get_data
    raise RuntimeError(f'DataLoader worker (pid(s) {pids_str}) exited unexpectedly') from e
RuntimeError: DataLoader worker (pid(s) 30128, 18392, 21084, 11432) exited unexpectedly


这个错误是典型的Windows系统下多进程编程问题，主要由于没有正确使用if __name__ == '__main__'保护主程序入口导致。以下是具体原因和解决方案：

问题原因分析： Windows多进程机制限制

Windows使用spawn方式创建子进程（而非Linux的fork方式），
子进程会重新导入主模块，导致循环执行代码，
需要if __name__ == '__main__'保护主程序入口
DataLoader多进程冲突：
设置了num_workers > 0时会启用多进程加载数据， 主程序中没有正确使用保护机制导致进程冲突