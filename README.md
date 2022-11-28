# GeoRegression

# Bug Record
## Fit problem
```python
def fit(self, X, y, coordinate_vector_list=None, weight_matrix=None):
    """
    关于多次Fit的时候，出现内存升高、CUP占用率不满的问题
    最后发现原因是因为并行的函数里面返回了estimator的引用，导致了这个问题。
    
    之后处理这种问题的时候，应该有每步详细的记录，这次记录的就很不详细。记录地详细才能更好地判断出错的原因。
    根据回忆记录一下：
    首先是把返回的estimator变成了None，发现内存还是升高。
    这里就很奇怪。不过当时是用tuple和zip进行返回结果的解构，难道是这个的影响？
    但是也不应该啊，那也应该是并行的任务结束之后，而不是运行时的内存变化。可能我当时把None又改回去了？
    然后把返回的estimator列表删除掉之后，内存就降下来了。这么一想，肯定就是它的原因啊。
    不过当时我也没想到，以为内存升高是正常现象，想的是怎么在运行之后把内存清理掉。
    后来还把返回结果从tuple/list转成了numpy的数组，想会不会释放了原有引用，内存自然就被清理了。
    估计确实起作用了，但是问题真的不知道在哪，不管了。
    
    在这之前，我还在尝试是不是tuple列表什么什么的原因。
    不过这里python的表现也很奇怪，明明下一次的会把上一次的结果给覆盖，内存就应该被清理掉了啊？
    看内存的占用，内存确实被清理掉了，但是为什么后面的CPU占用就占用不满了呢？？
    确实很奇怪哦。
    不过不管怎么样，确实应该详细记录各个操作的表现的。这种很麻烦、复杂、耗时的问题，就应该有清晰的实验结构才行。
    这个问题到底怎么导致的，就不去纠结了，因为这种多进程的问题确实都涉及到底层，无论是并行库的原理还是进程之间的调度。
    不过解决了就好...估计原因大致就是那个方向的。
    
    # 只要接受了返回的东西，就会不断变慢。要完全解除引用，要么隐式覆盖、要么显式Del掉。
    # self.parallel(delay_func_list)
    
    # Del掉返回的结果就可以了
    # del _
    
    所以最后的解决方法，其实就是把代码统统删掉，变成最简单的方式。
    下次需要注意并行任务的返回结果。因为它是多个进程执行，涉及到进程之间的数据交流，无论是传入还是返回开销都很大。
    这里肯定是把整个estimator都进行序列化，然后返回主进程了。
    """
    # Parallel run the job. return [(prediction, estimator), (), ...]
    parallel_result = Parallel(n_jobs)(
        delayed(fit_local_estimator)(
            estimator, X[neighbour_mask], y[neighbour_mask], local_x=x,
            sample_weight=row_weight[neighbour_mask],
            return_estimator=cache_estimator
        )
        for index, estimator, neighbour_mask, row_weight, x in
        zip(local_indices, estimator_list, neighbour_matrix, weight_matrix, X_predict)
        if index in local_indices
    )

    local_predict, local_estimator_list = list(zip(*parallel_result))


```
