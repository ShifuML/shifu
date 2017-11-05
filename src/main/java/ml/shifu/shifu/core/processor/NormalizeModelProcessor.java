/*
 * Copyright [2012-2014] PayPal Software Foundation
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
package ml.shifu.shifu.core.processor;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import ml.shifu.shifu.actor.AkkaSystemExecutor;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.shuffle.MapReduceShuffle;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.pig.PigExecutor;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;

import org.apache.commons.lang.StringUtils;
import org.apache.pig.tools.pigstats.JobStats;
import org.apache.pig.tools.pigstats.PigStats;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Normalize processor, scaling data
 */
public class NormalizeModelProcessor extends BasicModelProcessor implements Processor {

    private final static Logger log = LoggerFactory.getLogger(NormalizeModelProcessor.class);

    private boolean isToShuffleData = false;

    public NormalizeModelProcessor() {
        this(false);
    }

    public NormalizeModelProcessor(boolean isToShuffleData) {
        this.isToShuffleData = isToShuffleData;
    }

    /**
     * runner for normalization data
     */
    @Override
    public int run() throws Exception {
        log.info("Step Start: normalize");
        long start = System.currentTimeMillis();
        try {
            setUp(ModelStep.NORMALIZE);
            syncDataToHdfs(modelConfig.getDataSet().getSource());

            switch(modelConfig.getBasic().getRunMode()) {
                case DIST:
                case MAPRED:
                    runPigNormalize();

                    try {
                        autoCheckShuffleAndShuffleSize();
                    } catch (Exception e) {
                        log.warn(
                                "warn: exception in autho check shuffle size, exception can be ignored as no big impact",
                                e);
                    }

                    if(this.isToShuffleData) {
                        // shuffling normalized data, to make data random
                        MapReduceShuffle shuffler = new MapReduceShuffle(this.modelConfig);
                        shuffler.run(this.pathFinder.getNormalizedDataPath());
                    }

                    if(CommonUtils.isTreeModel(modelConfig.getAlgorithm())) {
                        runDataClean(this.isToShuffleData);
                    }
                    break;
                case LOCAL:
                    runAkkaNormalize();
                    break;
            }

            syncDataToHdfs(modelConfig.getDataSet().getSource());
            clearUp(ModelStep.NORMALIZE);
        } catch (Exception e) {
            log.error("Error:", e);
            return -1;
        }
        log.info("Step Finished: normalize with {} ms", (System.currentTimeMillis() - start));
        return 0;
    }

    private void autoCheckShuffleAndShuffleSize() throws IOException {
        ColumnConfig targetColumnConfig = CommonUtils.findTargetColumn(columnConfigList);
        Long totalCount = targetColumnConfig.getTotalCount();
        if(totalCount == null) {
            return;
        }
        // how many part-m-*.gz file in for gzip file, norm depends on how many gzip files
        int filePartCnt = ShifuFileUtils.getFilePartCount(this.pathFinder.getNormalizedDataPath(), SourceType.HDFS);
        // average count is over threshold, try to do shuffle to avoid big worker there
        if(filePartCnt > 0 && filePartCnt <= CommonConstants.PART_FILE_COUNT_THRESHOLD
                && totalCount * 1.0d / filePartCnt >= CommonConstants.MAX_RECORDS_PER_WORKER
                && ShifuFileUtils.isPartFileAllGzip(this.pathFinder.getNormalizedDataPath(), SourceType.HDFS)) {
            long shuffleSize = totalCount / CommonConstants.MAX_RECORDS_PER_WORKER;
            log.info("New shiffle size is {}.", shuffleSize);

            this.isToShuffleData = true;
            Integer shuffleSizeInteger = Environment.getInt(Constants.SHIFU_NORM_SHUFFLE_SIZE);
            if(shuffleSizeInteger == null) {
                Environment.setProperty(Constants.SHIFU_NORM_SHUFFLE_SIZE, shuffleSize + "");
            }
        }
    }

    /**
     * running akka normalize process
     * 
     * @throws IOException
     */
    private void runAkkaNormalize() throws IOException {
        SourceType sourceType = modelConfig.getDataSet().getSource();

        ShifuFileUtils.deleteFile(pathFinder.getNormalizedDataPath(), sourceType);
        ShifuFileUtils.deleteFile(pathFinder.getSelectedRawDataPath(), sourceType);

        List<Scanner> scanners = null;
        try {
            scanners = ShifuFileUtils.getDataScanners(
                    ShifuFileUtils.expandPath(modelConfig.getDataSetRawPath(), sourceType), sourceType);
        } catch (IOException e) {
            throw new ShifuException(ShifuErrorCode.ERROR_INPUT_NOT_FOUND, e, ", could not get input files "
                    + modelConfig.getDataSetRawPath());
        }

        if(scanners == null || scanners.size() == 0) {
            throw new ShifuException(ShifuErrorCode.ERROR_INPUT_NOT_FOUND, ", please check the data in "
                    + modelConfig.getDataSetRawPath() + " in " + sourceType);
        }

        AkkaSystemExecutor.getExecutor().submitNormalizeJob(modelConfig, columnConfigList, scanners);

        // release
        closeScanners(scanners);
    }

    /**
     * Running pig normalize process
     * 
     * @throws IOException
     *             any IO exception.
     */
    @SuppressWarnings("deprecation")
    private void runPigNormalize() throws IOException {
        SourceType sourceType = modelConfig.getDataSet().getSource();

        ShifuFileUtils.deleteFile(pathFinder.getNormalizedDataPath(), sourceType);
        ShifuFileUtils.deleteFile(pathFinder.getSelectedRawDataPath(), sourceType);

        Map<String, String> paramsMap = new HashMap<String, String>();
        paramsMap.put("sampleRate", modelConfig.getNormalizeSampleRate().toString());
        paramsMap.put("sampleNegOnly", ((Boolean) modelConfig.isNormalizeSampleNegOnly()).toString());
        paramsMap.put("delimiter", CommonUtils.escapePigString(modelConfig.getDataSetDelimiter()));

        String expressionsAsString = super.modelConfig.getSegmentFilterExpressionsAsString();
        Environment.getProperties().put("shifu.segment.expressions", expressionsAsString);

        try {
            String normPigPath = null;
            if(modelConfig.getNormalize().getIsParquet()) {
                if(modelConfig.getBasic().getPostTrainOn()) {
                    normPigPath = pathFinder.getScriptPath("scripts/NormalizeWithParquetAndPostTrain.pig");
                } else {
                    log.info("Post train is disabled by 'postTrainOn=false'.");
                    normPigPath = pathFinder.getScriptPath("scripts/NormalizeWithParquet.pig");
                }
            } else {
                if(modelConfig.getBasic().getPostTrainOn()) {
                    // this condition is for comment, no matter post train enabled or not, only norm results will be
                    // stored since new post train solution no need to prepare data
                }
                normPigPath = pathFinder.getScriptPath("scripts/Normalize.pig");
            }
            paramsMap.put(Constants.IS_COMPRESS, "true");
            paramsMap.put(Constants.IS_NORM_FOR_CLEAN, "false");
            PigExecutor.getExecutor().submitJob(modelConfig, normPigPath, paramsMap);

            Iterator<JobStats> iter = PigStats.get().getJobGraph().iterator();

            while(iter.hasNext()) {
                JobStats jobStats = iter.next();
                if(jobStats.getHadoopCounters() != null
                        && jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER) != null) {
                    long totalValidCount = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                            .getCounter("TOTAL_VALID_COUNT");
                    // If no basic record counter, check next one
                    if(totalValidCount == 0L) {
                        continue;
                    }
                    long invalidTagCount = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                            .getCounter("INVALID_TAG");

                    log.info("Total valid records {} after filtering, invalid tag records {}.", totalValidCount,
                            invalidTagCount);

                    if(totalValidCount > 0L && invalidTagCount * 1d / totalValidCount >= 0.8d) {
                        log.error("Too many invalid tags, please check you configuration on positive tags and negative tags.");
                    }
                }
                // only one pig job with such counters, break
                break;
            }

            if(StringUtils.isNotBlank(modelConfig.getValidationDataSetRawPath())) {
                ShifuFileUtils.deleteFile(pathFinder.getNormalizedValidationDataPath(), sourceType);
                paramsMap.put(Constants.IS_COMPRESS, "false");
                paramsMap.put(Constants.PATH_RAW_DATA, modelConfig.getValidationDataSetRawPath());
                paramsMap.put(Constants.PATH_NORMALIZED_DATA, pathFinder.getNormalizedValidationDataPath());
                PigExecutor.getExecutor().submitJob(modelConfig, normPigPath, paramsMap);
            }
        } catch (IOException e) {
            throw new ShifuException(ShifuErrorCode.ERROR_RUNNING_PIG_JOB, e);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

}
