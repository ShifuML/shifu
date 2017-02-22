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
package ml.shifu.shifu.guagua;

import java.io.IOException;
import java.lang.reflect.Constructor;

import ml.shifu.guagua.GuaguaRuntimeException;
import ml.shifu.guagua.hadoop.io.GuaguaWritableAdapter;
import ml.shifu.guagua.io.GuaguaFileSplit;
import ml.shifu.guagua.io.GuaguaRecordReader;

import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.TaskAttemptID;
import org.apache.pig.data.Tuple;

import parquet.filter.UnboundRecordFilter;
import parquet.filter2.compat.FilterCompat;
import parquet.filter2.compat.FilterCompat.Filter;
import parquet.filter2.predicate.FilterPredicate;
import parquet.hadoop.BadConfigurationException;
import parquet.hadoop.ParquetInputSplit;
import parquet.hadoop.ParquetRecordReader;
import parquet.hadoop.api.ReadSupport;
import parquet.hadoop.util.ConfigurationUtil;
import parquet.hadoop.util.SerializationUtil;
import parquet.pig.TupleReadSupport;

/**
 * {@link GuaguaParquetRecordReader} is a reader to read parquet format data to pig tuple format.
 * 
 * <p>
 * {@link GuaguaParquetRecordReader} is depending on pig and temperally to make it in shifu. In long term, this should
 * be migrated to Guagua.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class GuaguaParquetRecordReader implements
        GuaguaRecordReader<GuaguaWritableAdapter<LongWritable>, GuaguaWritableAdapter<Tuple>> {

    /**
     * key to configure the ReadSupport implementation
     */
    public static final String READ_SUPPORT_CLASS = "parquet.read.support.class";

    /**
     * key to configure the filter predicate
     */
    public static final String FILTER_PREDICATE = "parquet.private.read.filter.predicate";

    /**
     * key to configure the filter
     */
    public static final String UNBOUND_RECORD_FILTER = "parquet.read.filter";

    private ParquetRecordReader<Tuple> parquetRecordReader;

    private Configuration conf;

    public GuaguaParquetRecordReader() {
        this.conf = new Configuration();
    }

    public GuaguaParquetRecordReader(GuaguaFileSplit split) throws IOException {
        this(new Configuration(), split);
    }

    public GuaguaParquetRecordReader(Configuration conf, GuaguaFileSplit split) throws IOException {
        this.conf = conf;
        initialize(split);
    }

    private boolean isHadoop2() {
        try {
            Class.forName("org.apache.hadoop.mapreduce.task.MapContextImpl");
            return true;
        } catch (ClassNotFoundException e) {
            return false;
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.io.GuaguaRecordReader#initialize(ml.shifu.guagua.io.GuaguaFileSplit)
     */
    @Override
    public void initialize(GuaguaFileSplit split) throws IOException {
        ReadSupport<Tuple> readSupport = getReadSupportInstance(this.conf);
        this.parquetRecordReader = new ParquetRecordReader<Tuple>(readSupport, getFilter(this.conf));

        ParquetInputSplit parquetInputSplit = new ParquetInputSplit(new Path(split.getPath()), split.getOffset(),
                split.getOffset() + split.getLength(), split.getLength(), null, null);
        try {
            this.parquetRecordReader.initialize(parquetInputSplit, buildContext());
        } catch (InterruptedException e) {
            throw new GuaguaRuntimeException(e);
        }
    }

    /*
     * Build context through reflection to make sure code compatible between hadoop 1 and hadoop 2
     */
    private TaskAttemptContext buildContext() {
        TaskAttemptID id = null;
        TaskAttemptContext context = null;

        try {
            if(isHadoop2()) {
                Class<?> taskTypeClass = Class.forName("org.apache.hadoop.mapreduce.TaskType");
                Constructor<TaskAttemptID> constructor = TaskAttemptID.class.getDeclaredConstructor(String.class,
                        Integer.TYPE, taskTypeClass, Integer.TYPE, Integer.TYPE);
                id = constructor.newInstance("mock", -1, fromEnumConstantName(taskTypeClass, "MAP"), -1, -1);
                Constructor<?> contextConstructor = Class.forName(
                        "org.apache.hadoop.mapreduce.task.TaskAttemptContextImpl").getDeclaredConstructor(
                        Configuration.class, TaskAttemptID.class);
                context = (TaskAttemptContext) contextConstructor.newInstance(this.conf, id);
            } else {
                Constructor<TaskAttemptID> constructor = TaskAttemptID.class.getDeclaredConstructor(String.class,
                        Integer.TYPE, Boolean.TYPE, Integer.TYPE, Integer.TYPE);
                constructor.setAccessible(true);
                id = constructor.newInstance("mock", -1, false, -1, -1);
                Constructor<?> contextConstructor = Class.forName("org.apache.hadoop.mapreduce.TaskAttemptContext")
                        .getDeclaredConstructor(Configuration.class, TaskAttemptID.class);
                context = (TaskAttemptContext) contextConstructor.newInstance(this.conf, id);
            }
        } catch (Throwable e) {
            throw new GuaguaRuntimeException(e);
        }
        return context;
    }

    private static FilterPredicate getFilterPredicate(Configuration configuration) {
        try {
            return SerializationUtil.readObjectFromConfAsBase64(FILTER_PREDICATE, configuration);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /*
     * Returns a non-null Filter, which is a wrapper around either a
     * FilterPredicate, an UnboundRecordFilter, or a no-op filter.
     */
    public static Filter getFilter(Configuration conf) {
        return FilterCompat.get(getFilterPredicate(conf), getUnboundRecordFilterInstance(conf));
    }

    private static UnboundRecordFilter getUnboundRecordFilterInstance(Configuration configuration) {
        Class<?> clazz = ConfigurationUtil.getClassFromConfig(configuration, UNBOUND_RECORD_FILTER,
                UnboundRecordFilter.class);
        if(clazz == null) {
            return null;
        }

        try {
            UnboundRecordFilter unboundRecordFilter = (UnboundRecordFilter) clazz.newInstance();

            if(unboundRecordFilter instanceof Configurable) {
                ((Configurable) unboundRecordFilter).setConf(configuration);
            }

            return unboundRecordFilter;
        } catch (InstantiationException e) {
            throw new BadConfigurationException("could not instantiate unbound record filter class", e);
        } catch (IllegalAccessException e) {
            throw new BadConfigurationException("could not instantiate unbound record filter class", e);
        }
    }

    /*
     * Return read support instance
     * 
     * @param configuration
     * to find the configuration for the read support
     * 
     * @return the configured read support
     */
    @SuppressWarnings("unchecked")
    public static <T> ReadSupport<T> getReadSupportInstance(Configuration configuration) {
        return getReadSupportInstance((Class<? extends ReadSupport<T>>) getReadSupportClass(configuration));
    }

    public static Class<?> getReadSupportClass(Configuration configuration) {
        Class<?> clazz = ConfigurationUtil.getClassFromConfig(configuration, READ_SUPPORT_CLASS, ReadSupport.class);
        if(clazz == null) {
            clazz = TupleReadSupport.class;
        }
        return clazz;
    }

    /*
     * Return read support instance
     * 
     * @param readSupportClass
     * to instantiate
     * 
     * @return the configured read support
     */
    static <T> ReadSupport<T> getReadSupportInstance(Class<? extends ReadSupport<T>> readSupportClass) {
        try {
            return readSupportClass.newInstance();
        } catch (InstantiationException e) {
            throw new BadConfigurationException("could not instantiate read support class", e);
        } catch (IllegalAccessException e) {
            throw new BadConfigurationException("could not instantiate read support class", e);
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.io.GuaguaRecordReader#nextKeyValue()
     */
    @Override
    public boolean nextKeyValue() throws IOException {
        try {
            return this.parquetRecordReader.nextKeyValue();
        } catch (InterruptedException e) {
            throw new GuaguaRuntimeException(e);
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.io.GuaguaRecordReader#getCurrentKey()
     */
    @Override
    public GuaguaWritableAdapter<LongWritable> getCurrentKey() {
        return null;
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.io.GuaguaRecordReader#getCurrentValue()
     */
    @Override
    public GuaguaWritableAdapter<Tuple> getCurrentValue() {
        try {
            return new GuaguaWritableAdapter<Tuple>(this.parquetRecordReader.getCurrentValue());
        } catch (IOException e) {
            throw new GuaguaRuntimeException(e);
        } catch (InterruptedException e) {
            throw new GuaguaRuntimeException(e);
        }
    }

    @SuppressWarnings("rawtypes")
    private static Enum fromEnumConstantName(Class<?> enumClass, String constantName) {
        Object[] enumConstants = enumClass.getEnumConstants();
        for(Object t: enumConstants) {
            if(((java.lang.Enum<?>) t).name().equals(constantName)) {
                return (Enum) t;
            }
        }
        return null;
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.io.GuaguaRecordReader#close()
     */
    @Override
    public void close() throws IOException {
        this.parquetRecordReader.close();
    }

}
