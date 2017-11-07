/*
 * Copyright [2013-2017] PayPal Software Foundation
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
package ml.shifu.shifu.core.correlation;

import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.hadoop.classification.InterfaceAudience;
import org.apache.hadoop.classification.InterfaceStability;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.OutputCommitter;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.StatusReporter;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.TaskAttemptID;
import org.apache.hadoop.util.ReflectionUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Copy from MultithreadedMapper to do some customization. Merge mapper output results and then write to reducer.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
@InterfaceAudience.Public
@InterfaceStability.Stable
public class FastCorrelationMultithreadedMapper extends Mapper<LongWritable, Text, IntWritable, CorrelationWritable> {

    private final static Logger LOG = LoggerFactory.getLogger(FastCorrelationMultithreadedMapper.class);
    public static String NUM_THREADS = "mapreduce.mapper.multithreadedmapper.threads";
    public static String MAP_CLASS = "mapreduce.mapper.multithreadedmapper.mapclass";

    private Class<? extends Mapper<LongWritable, Text, IntWritable, CorrelationWritable>> mapClass;
    private Context outer;
    private List<MapRunner> runners;

    private Map<Integer, CorrelationWritable> finalCorrelationMap;

    /**
     * Output key cache to avoid new operation.
     */
    private IntWritable outputKey;

    /**
     * The number of threads in the thread pool that will run the map function.
     * 
     * @param job
     *            the job
     * @return the number of threads
     */
    public static int getNumberOfThreads(JobContext job) {
        return job.getConfiguration().getInt(NUM_THREADS, 10);
    }

    /**
     * Set the number of threads in the pool for running maps.
     * 
     * @param job
     *            the job to modify
     * @param threads
     *            the new number of threads
     */
    public static void setNumberOfThreads(Job job, int threads) {
        job.getConfiguration().setInt(NUM_THREADS, threads);
    }

    /**
     * Get the application's mapper class.
     * 
     * @param <K1>
     *            the map's input key type
     * @param <V1>
     *            the map's input value type
     * @param <K2>
     *            the map's output key type
     * @param <V2>
     *            the map's output value type
     * @param job
     *            the job
     * @return the mapper class to run
     */
    @SuppressWarnings("unchecked")
    public static <K1, V1, K2, V2> Class<Mapper<K1, V1, K2, V2>> getMapperClass(JobContext job) {
        return (Class<Mapper<K1, V1, K2, V2>>) job.getConfiguration().getClass(MAP_CLASS, Mapper.class);
    }

    /**
     * Set the application's mapper class.
     * 
     * @param <K1>
     *            the map input key type
     * @param <V1>
     *            the map input value type
     * @param <K2>
     *            the map output key type
     * @param <V2>
     *            the map output value type
     * @param job
     *            the job to modify
     * @param cls
     *            the class to use as the mapper
     */
    public static <K1, V1, K2, V2> void setMapperClass(Job job, Class<? extends Mapper<K1, V1, K2, V2>> cls) {
        if(FastCorrelationMultithreadedMapper.class.isAssignableFrom(cls)) {
            throw new IllegalArgumentException("Can't have recursive " + "MultithreadedMapper instances.");
        }
        job.getConfiguration().setClass(MAP_CLASS, cls, Mapper.class);
    }

    /**
     * Run the application's maps using a thread pool.
     */
    @Override
    public void run(Context context) throws IOException, InterruptedException {
        outer = context;
        this.finalCorrelationMap = new HashMap<Integer, CorrelationWritable>();

        int numberOfThreads = getNumberOfThreads(context);
        mapClass = getMapperClass(context);
        if(LOG.isDebugEnabled()) {
            LOG.debug("Configuring multithread runner to use " + numberOfThreads + " threads");
        }

        runners = new ArrayList<MapRunner>(numberOfThreads);
        for(int i = 0; i < numberOfThreads; ++i) {
            MapRunner thread = new MapRunner(context);
            thread.start();
            runners.add(i, thread);
        }
        for(int i = 0; i < numberOfThreads; ++i) {
            MapRunner thread = runners.get(i);
            thread.join();
            Throwable th = thread.throwable;
            if(th != null) {
                if(th instanceof IOException) {
                    throw (IOException) th;
                } else if(th instanceof InterruptedException) {
                    throw (InterruptedException) th;
                } else {
                    throw new RuntimeException(th);
                }
            }
        }

        outputKey = new IntWritable();

        // after all sub mapper completed, finalCorrelationMap includes global results and send them to reducer.
        // send to reducer with only one merged copy no matter how many threads
        synchronized(outer) {
            for(Entry<Integer, CorrelationWritable> entry: finalCorrelationMap.entrySet()) {
                outputKey.set(entry.getKey());
                context.write(outputKey, entry.getValue());
            }
        }
    }

    private class SubMapRecordReader extends RecordReader<LongWritable, Text> {
        private LongWritable key;
        private Text value;
        private Configuration conf;

        @Override
        public void close() throws IOException {
        }

        @Override
        public float getProgress() throws IOException, InterruptedException {
            return 0;
        }

        @Override
        public void initialize(InputSplit split, TaskAttemptContext context) throws IOException, InterruptedException {
            conf = context.getConfiguration();
        }

        @Override
        public boolean nextKeyValue() throws IOException, InterruptedException {
            synchronized(outer) {
                if(!outer.nextKeyValue()) {
                    return false;
                }
                key = ReflectionUtils.copy(outer.getConfiguration(), outer.getCurrentKey(), key);
                value = ReflectionUtils.copy(conf, outer.getCurrentValue(), value);
                return true;
            }
        }

        public LongWritable getCurrentKey() {
            return key;
        }

        @Override
        public Text getCurrentValue() {
            return value;
        }
    }

    private class SubMapRecordWriter extends RecordWriter<IntWritable, CorrelationWritable> {

        @Override
        public void close(TaskAttemptContext context) throws IOException, InterruptedException {
        }

        @Override
        public void write(IntWritable key, CorrelationWritable value) throws IOException, InterruptedException {
            synchronized(outer) {
                // replace outer.write by merge to one global corrMap
                CorrelationWritable cw = FastCorrelationMultithreadedMapper.this.finalCorrelationMap.get(key.get());
                if(cw == null) {
                    cw = value;
                } else {
                    cw.combine(value);
                }
                FastCorrelationMultithreadedMapper.this.finalCorrelationMap.put(key.get(), cw);
            }
        }
    }

    private class SubMapStatusReporter extends StatusReporter {

        @Override
        public Counter getCounter(Enum<?> name) {
            return outer.getCounter(name);
        }

        @Override
        public Counter getCounter(String group, String name) {
            return outer.getCounter(group, name);
        }

        @Override
        public void progress() {
            outer.progress();
        }

        @Override
        public void setStatus(String status) {
            outer.setStatus(status);
        }

        public float getProgress() {
            try {
                Method method = outer.getClass().getDeclaredMethod("getProgress", new Class[] {});
                if(method != null) {
                    return (Float) (method.invoke(outer, new Object[] {}));
                }
            } catch (Throwable e) {
                return 0f;
            }
            return 0f;
        }
    }

    private class MapRunner extends Thread {
        private Mapper<LongWritable, Text, IntWritable, CorrelationWritable> mapper;
        private Context subcontext;
        private Throwable throwable;
        private RecordReader<LongWritable, Text> reader = new SubMapRecordReader();

        MapRunner(Context context) throws IOException, InterruptedException {
            mapper = ReflectionUtils.newInstance(mapClass, context.getConfiguration());
            subcontext = createSubContext(context);
            reader.initialize(context.getInputSplit(), context);
        }

        private Context createSubContext(Context context) {
            boolean isHadoop2 = false;
            Class<?> mapContextImplClazz = null;
            try {
                mapContextImplClazz = Class.forName("org.apache.hadoop.mapreduce.task.MapContextImpl");
                isHadoop2 = true;
            } catch (ClassNotFoundException e) {
                isHadoop2 = false;
            }

            if(mapContextImplClazz == null) {
                isHadoop2 = false;
            }

            try {
                if(isHadoop2) {
                    return createSubContextForHadoop2(context, mapContextImplClazz);
                } else {
                    return createSubContextForHadoop1(context);
                }
            } catch (Throwable t) {
                throw new RuntimeException(t);
            }
        }

        @SuppressWarnings("unchecked")
        private Context createSubContextForHadoop2(Context context, Class<?> mapContextImplClazz)
                throws NoSuchMethodException, InstantiationException, IllegalAccessException,
                InvocationTargetException, ClassNotFoundException {
            Constructor<?> constructor = mapContextImplClazz.getDeclaredConstructor(Configuration.class,
                    TaskAttemptID.class, RecordReader.class, RecordWriter.class, OutputCommitter.class,
                    StatusReporter.class, InputSplit.class);
            constructor.setAccessible(true);
            Object mapContext = constructor.newInstance(outer.getConfiguration(), outer.getTaskAttemptID(), reader,
                    new SubMapRecordWriter(), context.getOutputCommitter(), new SubMapStatusReporter(),
                    outer.getInputSplit());
            Class<?> wrappedMapperClazz = Class.forName("org.apache.hadoop.mapreduce.lib.map.WrappedMapper");
            Object wrappedMapper = wrappedMapperClazz.newInstance();
            Method method = wrappedMapperClazz.getDeclaredMethod("getMapContext",
                    Class.forName("org.apache.hadoop.mapreduce.MapContext"));
            return (Context) (method.invoke(wrappedMapper, mapContext));
        }

        @SuppressWarnings("unchecked")
        private Context createSubContextForHadoop1(Context context) throws NoSuchMethodException,
                InstantiationException, IllegalAccessException, InvocationTargetException {
            Constructor<?> constructor = Context.class.getDeclaredConstructor(Mapper.class, Configuration.class,
                    TaskAttemptID.class, RecordReader.class, RecordWriter.class, OutputCommitter.class,
                    StatusReporter.class, InputSplit.class);
            constructor.setAccessible(true);
            return (Context) constructor.newInstance(mapper, outer.getConfiguration(), outer.getTaskAttemptID(),
                    reader, new SubMapRecordWriter(), context.getOutputCommitter(), new SubMapStatusReporter(),
                    outer.getInputSplit());
        }

        @SuppressWarnings("unused")
        public Throwable getThrowable() {
            return throwable;
        }

        @Override
        public void run() {
            try {
                mapper.run(subcontext);
            } catch (Throwable ie) {
                throwable = ie;
            } finally {
                try {
                    reader.close();
                } catch (IOException ignore) {
                }
            }
        }
    }

}
