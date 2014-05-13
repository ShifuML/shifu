/**
 * Copyright [2012-2013] eBay Software Foundation
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
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.collections.CollectionUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.shifu.actor.AkkaSystemExecutor;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ColumnConfig.ColumnType;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.pig.PigExecutor;
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
 * </p>
 *
 */
public class StatsModelProcessor extends BasicModelProcessor implements Processor{

    private final static Logger log = LoggerFactory.getLogger(StatsModelProcessor.class);

    private static ObjectMapper jsonMapper = new ObjectMapper();
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
     *
     */
    private void runAkkaStats(){
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
    private void runPigStats() throws IOException{
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
    private void scanStatsResult(Scanner scanner) throws IOException {
        while(scanner.hasNextLine()) {
            String[] raw = scanner.nextLine().trim().split("\\|");

            if(raw.length == 1) {
                continue;
            }

            int columnNum = Integer.parseInt(raw[0]);

            ColumnConfig config = jsonMapper.readValue(raw[1], ColumnConfig.class);

            columnConfigList.set(columnNum, config);


        }
    }

}
