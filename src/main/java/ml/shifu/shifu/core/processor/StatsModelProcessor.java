/**
 * Copyright [2012-2014] eBay Software Foundation
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

import ml.shifu.shifu.actor.AkkaSystemExecutor;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ColumnConfig.ColumnType;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.pig.PigExecutor;
import ml.shifu.shifu.udf.CalculateStatsUDF;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.collections.CollectionUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

/**
 * statistics, max/min/avg/std for each column dataset if it's numerical
 */
public class StatsModelProcessor extends BasicModelProcessor implements Processor {

    private final static Logger log = LoggerFactory.getLogger(StatsModelProcessor.class);

    /**
     * runner for statistics
     */
    @Override
    public int run() throws Exception {
        log.info("Step Start: stats");
        setUp(ModelStep.STATS);

        syncDataToHdfs(modelConfig.getDataSet().getSource());

        if(modelConfig.isMapReduceRunMode()) {
            runPigStats();
        } else if(modelConfig.isLocalRunMode()) {
            runAkkaStats();
        } else {
            throw new ShifuException(ShifuErrorCode.ERROR_UNSUPPORT_MODE);
        }
        clearUp(ModelStep.STATS);

        log.info("Step Finished: stats");
        return 0;
    }

    /**
     * run akka stats
     */
    private void runAkkaStats() {
        List<Scanner> scanners = null;

        try {
            SourceType sourceType = modelConfig.getDataSet().getSource();
            // the bug is caused when merging code? please take care
            scanners = ShifuFileUtils.getDataScanners(
                    ShifuFileUtils.expandPath(modelConfig.getDataSetRawPath(), sourceType), sourceType);
        } catch (IOException e) {
            throw new ShifuException(ShifuErrorCode.ERROR_INPUT_NOT_FOUND, e);
        }

        if(CollectionUtils.isEmpty(scanners)) {
            throw new ShifuException(ShifuErrorCode.ERROR_INPUT_NOT_FOUND,
                    ", please check your data and start from init");
        }

        log.info("Num of Scanners: " + scanners.size());

        AkkaSystemExecutor.getExecutor().submitStatsCalJob(modelConfig, columnConfigList, scanners);

        // release
        closeScanners(scanners);
    }

    /**
     * run pig stats
     * 
     * @throws IOException
     */
    private void runPigStats() throws IOException {
        log.info("delete historical pre-train data");

        ShifuFileUtils.deleteFile(pathFinder.getPreTrainingStatsPath(), modelConfig.getDataSet().getSource());
        Map<String, String> paramsMap = new HashMap<String, String>();
        paramsMap.put("delimiter", CommonUtils.escapePigString(modelConfig.getDataSetDelimiter()));

        // execute pig job
        try {
            PigExecutor.getExecutor().submitJob(modelConfig,
                    pathFinder.getAbsolutePath("scripts/PreTrainingStats.pig"), paramsMap);
        } catch (IOException e) {
            throw new ShifuException(ShifuErrorCode.ERROR_RUNNING_PIG_JOB, e);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }

        // sync Down
        log.info("Updating ColumnConfig with stats...");
        // update column config
        updateColumnConfigWithPreTrainingStats();
        // save it to local/hdfs
        saveColumnConfigList();
    }

    /**
     * update the max/min/mean/std/binning information from stats step
     * 
     * @throws IOException
     */
    public void updateColumnConfigWithPreTrainingStats() throws IOException {
        List<Scanner> scanners = ShifuFileUtils.getDataScanners(pathFinder.getPreTrainingStatsPath(), modelConfig
                .getDataSet().getSource());
        for(Scanner scanner: scanners) {
            scanStatsResult(scanner);
        }

        // release
        closeScanners(scanners);
    }

    /**
     * Scan the stats result and save them into column configure
     * 
     * @param scanner
     */
    private void scanStatsResult(Scanner scanner) {
        while(scanner.hasNextLine()) {
            String[] raw = scanner.nextLine().trim().split("\\|");

            if(raw.length == 1) {
                continue;
            }

            int columnNum = Integer.parseInt(raw[0]);
            try {
                ColumnConfig config = this.columnConfigList.get(columnNum);

                if(config.isCategorical()) {
                    config.setBinCategory(CommonUtils.stringToStringList(raw[1], CalculateStatsUDF.CATEGORY_VAL_SEPARATOR));
                } else {
                    config.setBinBoundary(CommonUtils.stringToDoubleList(raw[1]));
                }
                config.setBinCountNeg(CommonUtils.stringToIntegerList(raw[2]));
                config.setBinCountPos(CommonUtils.stringToIntegerList(raw[3]));
                // config.setBinAvgScore(CommonUtils.stringToIntegerList(raw[4]));
                config.setBinPosCaseRate(CommonUtils.stringToDoubleList(raw[5]));
                config.setBinLength(config.getBinCountNeg().size());
                config.setKs(Double.valueOf(raw[6]));
                config.setIv(Double.valueOf(raw[7]));
                config.setMax(Double.valueOf(raw[8]));
                config.setMin(Double.valueOf(raw[9]));
                config.setMean(Double.valueOf(raw[10]));
                config.setStdDev(Double.valueOf(raw[11]));

                // magic?
                if(raw[12].equals("N")) {
                    config.setColumnType(ColumnType.N);
                } else {
                    config.setColumnType(ColumnType.C);
                }

                config.setMedian(Double.valueOf(raw[13]));

                config.setMissingCnt(Long.valueOf(raw[14]));
                config.setTotalCount(Long.valueOf(raw[15]));
                config.setMissingPercentage(Double.valueOf(raw[16]));
                
                config.setBinWeightedNeg(CommonUtils.stringToDoubleList(raw[17]));
                config.setBinWeightedPos(CommonUtils.stringToDoubleList(raw[18]));

            } catch (Exception e) {
            	log.error("Fail to process following column : {} name: {}", columnNum, this.columnConfigList.get(columnNum).getColumnName());
                
            	continue;
            }
        }
    }

}
