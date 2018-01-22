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
package ml.shifu.shifu.udf;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.binning.AbstractBinning;
import ml.shifu.shifu.core.binning.DynamicBinning;
import ml.shifu.shifu.core.binning.obj.NumBinInfo;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.HDFSUtils;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.io.Charsets;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.CompressionCodecFactory;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.util.UDFContext;

/**
 * Created by zhanhu on 7/6/16.
 */
public class DynamicBinningUDF extends AbstractTrainerUDF<Tuple> {

    private Map<Integer, String> smallBinsMap;

    private String smallBinsPath;

    public DynamicBinningUDF(String source, String pathModelConfig, String pathColumnConfig, String smallBinsPath)
            throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
        this.smallBinsPath = smallBinsPath;
    }

    @Override
    public Tuple exec(Tuple input) throws IOException {
        // move initialization from constructor to be here because of Pig UDF will be called in client which will cause
        // OOM in there
        if(smallBinsMap == null) {
            smallBinsMap = new HashMap<Integer, String>();
            initSmallBinMap();
        }

        if(input == null || input.size() != 1) {
            return null;
        }

        Integer columnId = null;
        ColumnConfig columnConfig = null;
        String binsData = null;

        Set<String> missingValSet = new HashSet<String>(super.modelConfig.getMissingOrInvalidValues());
        List<NumBinInfo> binInfoList = null;

        DataBag columnDataBag = (DataBag) input.get(0);
        Iterator<Tuple> iterator = columnDataBag.iterator();
        while(iterator.hasNext()) {
            Tuple tuple = iterator.next();
            if(columnId == null) {
                columnId = (Integer) tuple.get(0);

                // for filter expansions
                if(columnId >= super.columnConfigList.size()) {
                    int newColumnId = columnId % super.columnConfigList.size();
                    columnConfig = super.columnConfigList.get(newColumnId);
                } else {
                    columnConfig = super.columnConfigList.get(columnId);
                }

                String smallBins = smallBinsMap.get(columnId);
                if(columnConfig.isCategorical()) {
                    binsData = smallBins;
                    break;
                } else {
                    binInfoList = NumBinInfo.constructNumBinfo(smallBins, AbstractBinning.FIELD_SEPARATOR);
                }
            }

            String val = (String) tuple.get(1);
            Boolean isPositiveInst = (Boolean) tuple.get(2);

            if(missingValSet.contains(val)) {
                continue;
            }

            Double d = null;

            try {
                d = Double.valueOf(val);
            } catch (Exception e) {
                // illegal number, just skip it
                continue;
            }

            NumBinInfo numBinInfo = binaryLocate(binInfoList, d);
            if(numBinInfo != null) {
                numBinInfo.incInstCnt(isPositiveInst);
            }
        }

        if(binsData == null && CollectionUtils.isNotEmpty(binInfoList)) {
            int maxNumBin = modelConfig.getStats().getMaxNumBin();
            if(maxNumBin <= 0) {
                maxNumBin = 1024;
            }
            DynamicBinning dynamicBinning = new DynamicBinning(binInfoList, maxNumBin);
            List<Double> binFields = dynamicBinning.getDataBin();
            binsData = StringUtils.join(binFields, CalculateStatsUDF.CATEGORY_VAL_SEPARATOR);
        }

        Tuple output = TupleFactory.getInstance().newTuple(2);

        output.set(0, columnId);
        output.set(1, binsData);

        return output;
    }

    private void initSmallBinMap() throws IOException, FileNotFoundException {
        long start = System.currentTimeMillis();
        Configuration jobConf = UDFContext.getUDFContext().getJobConf();
        // only load nesscary part files
        // this assumes dynamic binning and small bin job has the same reducers, not good but works
        String partFile = smallBinsPath + File.separator + "part-*-*" + jobConf.get("mapreduce.task.partition") + "*";

        FileStatus[] fileStatus = ShifuFileUtils.getFilePartStatus(partFile, SourceType.HDFS);
        if(fileStatus.length < 1) {
            throw new FileNotFoundException("small bin part file not found");
        }
        Path smallBinPartFilePath = fileStatus[0].getPath();

        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(SourceType.HDFS);
        CompressionCodecFactory compressionFactory = new CompressionCodecFactory(jobConf);
        BufferedReader reader = null;

        try {
            CompressionCodec codec = compressionFactory.getCodec(smallBinPartFilePath);
            InputStream is = null;
            if(codec != null) {
                is = codec.createInputStream(fs.open(smallBinPartFilePath));
            } else {
                is = fs.open(smallBinPartFilePath);
            }

            reader = new BufferedReader(new InputStreamReader(is, Charsets.toCharset("UTF-8")));
            String line = reader.readLine();
            while(line != null) {
                String[] fields = StringUtils.split(line, '\u0007');
                if(fields.length == 2) {
                    smallBinsMap.put(Integer.parseInt(fields[0]), fields[1]);
                }
                line = reader.readLine();
            }
        } finally {
            IOUtils.closeQuietly(reader);
        }
        log.info("smallBinPartFilePath is " + smallBinPartFilePath + " initialized in"
                + (System.currentTimeMillis() - start) + "ms.");
    }

    @SuppressWarnings("unused")
    private String readByColumnId(String smallBinsPath, Integer columnId) throws IOException {
        long start = System.currentTimeMillis();
        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(SourceType.HDFS);
        FileStatus[] fileStatsArr = ShifuFileUtils.getFilePartStatus(smallBinsPath, SourceType.HDFS);

        CompressionCodecFactory compressionFactory = new CompressionCodecFactory(HDFSUtils.getConf());
        for(FileStatus fileStatus: fileStatsArr) {
            BufferedReader reader = null;
            try {
                CompressionCodec codec = compressionFactory.getCodec(fileStatus.getPath());
                InputStream is = null;
                if(codec != null) {
                    is = codec.createInputStream(fs.open(fileStatus.getPath()));
                } else {
                    is = fs.open(fileStatus.getPath());
                }
                reader = new BufferedReader(new InputStreamReader(is, Charsets.toCharset("UTF-8")));
                String line = reader.readLine();
                while(line != null) {
                    String[] fields = StringUtils.split(line, '\u0007');
                    if(fields.length == 2) {
                        if(columnId == Integer.parseInt(fields[0])) {
                            log.info("Read small bins with columnId " + columnId + " in time "
                                    + (System.currentTimeMillis() - start) + "ms.");
                            return fields[1];
                        }
                    }
                    line = reader.readLine();
                }
            } finally {
                IOUtils.closeQuietly(reader);
            }
        }

        throw new RuntimeException("No such column in small bins output, please check small bin pig job");
    }

    public NumBinInfo binaryLocate(List<NumBinInfo> binInfoList, Double d) {
        int left = 0;
        int right = binInfoList.size() - 1;

        while(left <= right) {
            int middle = (left + right) / 2;
            NumBinInfo binInfo = binInfoList.get(middle);
            if(d >= binInfo.getLeftThreshold() && d < binInfo.getRightThreshold()) {
                return binInfo;
            } else if(d >= binInfo.getRightThreshold()) {
                left = middle + 1;
            } else if(d < binInfo.getLeftThreshold()) {
                right = middle - 1;
            } else {
                return null;
            }
        }

        return null;
    }
}
