/*
 * Copyright [2013-2015] PayPal Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.executor;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import ml.shifu.shifu.util.Environment;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by zhanhu on 12/12/16.
 */
public class ExecutorManager<T> {

    private static Logger LOG = LoggerFactory.getLogger(ExecutorManager.class);

    private ExecutorService executorService = null;

    public ExecutorManager() {
        this(Environment.getInt("shifu.combo.thread.parallel", 10));
    }

    public ExecutorManager(int threadPoolSize) {
        this.executorService = Executors.newFixedThreadPool(threadPoolSize);
    }

    @SuppressWarnings("rawtypes")
    public void submitTasksAndWaitFinish(List<Runnable> tasks) {
        List<Future<?>> futureList = new ArrayList<Future<?>>(tasks.size());
        for ( Runnable task : tasks ) {
            Future<?> future = executorService.submit(task);
            futureList.add(future);
        }

        for ( Future future : futureList ) {
            try {
                future.get();
            } catch (InterruptedException e) {
                LOG.error("Error occurred, when waiting task to finish.", e);
            } catch (ExecutionException e) {
                LOG.error("Error occurred, when waiting task to finish.", e);
            }
        }

        return;
    }

    public List<T> submitTasksAndWaitResults(List<Callable<T>> tasks) {
        List<T> results = new ArrayList<T>();

        List<Future<T>> futureList = new ArrayList<Future<T>>(tasks.size());
        for ( Callable<T> task : tasks ) {
            Future<T> future = executorService.submit(task);
            futureList.add(future);
        }

        for ( Future<T> future : futureList ) {
            try {
                results.add(future.get());
            } catch (InterruptedException e) {
                LOG.error("Error occurred, when waiting task to finish.", e);
            } catch (ExecutionException e) {
                LOG.error("Error occurred, when waiting task to finish.", e);
            }
        }

        return results;
    }

    public void graceShutDown() {
        this.executorService.shutdown();
        try {
            this.executorService.awaitTermination(Integer.MAX_VALUE, TimeUnit.SECONDS);
        } catch ( Exception e ) {
            LOG.error("Error occurred, when waiting task to finish.", e);
        }
    }

    public void forceShutDown() {
        this.executorService.shutdownNow();
        try {
            this.executorService.awaitTermination(2, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

}
