package ml.shifu.shifu.executor;

import ml.shifu.shifu.util.Environment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

/**
 * Created by zhanhu on 12/12/16.
 */
public class ExecutorManager {

    private static Logger LOG = LoggerFactory.getLogger(ExecutorManager.class);

    private ExecutorService executorService = Executors.newFixedThreadPool(
            Environment.getInt("shifu.combo.thread.parallel", 10)
    );

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
    }
}
