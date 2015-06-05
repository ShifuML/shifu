/**
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

import ml.shifu.shifu.actor.AkkaSystemExecutor;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.pig.PigExecutor;
import ml.shifu.shifu.util.CommonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.*;

/**
 * Post train processor, update the avg score
 */
public class PostTrainModelProcessor extends BasicModelProcessor implements Processor {

    /**
     * log object
     */
    private final static Logger log = LoggerFactory.getLogger(PostTrainModelProcessor.class);

    /**
     * runner for post train
     */
    @Override
    public int run() throws Exception {
        log.info("Step Start: posttrain");
        long start = System.currentTimeMillis();

        setUp(ModelStep.POSTTRAIN);
        syncDataToHdfs(modelConfig.getDataSet().getSource());

        if(modelConfig.isMapReduceRunMode()) {
            runPigPostTrain();
        } else if(modelConfig.isLocalRunMode()) {
            runAkkaPostTrain();
        } else {
            log.error("Invalid RunMode Setting!");
        }

        clearUp(ModelStep.POSTTRAIN);
        log.info("Step Finished: posttrain with {} ms", (System.currentTimeMillis() - start));

        return 0;
    }

    /**
     * run pig post train
     * 
     * @throws IOException
     */
    private void runPigPostTrain() throws IOException {
        SourceType sourceType = modelConfig.getDataSet().getSource();

        ShifuFileUtils.deleteFile(pathFinder.getTrainScoresPath(), sourceType);
        ShifuFileUtils.deleteFile(pathFinder.getBinAvgScorePath(), sourceType);

        // prepare special parameters and execute pig
        Map<String, String> paramsMap = new HashMap<String, String>();
        paramsMap.put("pathHeader", modelConfig.getHeaderPath());
        paramsMap.put("pathDelimiter", CommonUtils.escapePigString(modelConfig.getHeaderDelimiter()));
        paramsMap.put("delimiter", CommonUtils.escapePigString(modelConfig.getDataSetDelimiter()));

        try {
            PigExecutor.getExecutor().submitJob(modelConfig, pathFinder.getAbsolutePath("scripts/PostTrain.pig"),
                    paramsMap);
        } catch (IOException e) {
            throw new ShifuException(ShifuErrorCode.ERROR_RUNNING_PIG_JOB, e);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }

        // Sync Down
        columnConfigList = updateColumnConfigWithBinAvgScore(columnConfigList);
        saveColumnConfigListAndColumnStats();
    }

    /**
     * run akka post train
     * 
     * @throws IOException
     */
    private void runAkkaPostTrain() throws IOException {
        SourceType sourceType = modelConfig.getDataSet().getSource();

        List<Scanner> scanners = ShifuFileUtils.getDataScanners(pathFinder.getSelectedRawDataPath(sourceType),
                sourceType);

        log.info("Num of Scanners: " + scanners.size());
        AkkaSystemExecutor.getExecutor().submitPostTrainJob(modelConfig, columnConfigList, scanners);

        closeScanners(scanners);
    }

    /**
     * read the binary average score and update them into column list
     * 
     * @param columnConfigList
     * @return
     * @throws IOException
     */
    private List<ColumnConfig> updateColumnConfigWithBinAvgScore(List<ColumnConfig> columnConfigList)
            throws IOException {
        List<Scanner> scanners = ShifuFileUtils.getDataScanners(pathFinder.getBinAvgScorePath(), modelConfig
                .getDataSet().getSource());

        // CommonUtils.getDataScanners(pathFinder.getBinAvgScorePath(), modelConfig.getDataSet().getSource());
        for(Scanner scanner: scanners) {
            while(scanner.hasNextLine()) {
                List<Integer> scores = new ArrayList<Integer>();
                String[] raw = scanner.nextLine().split("\\|");
                int columnNum = Integer.parseInt(raw[0]);
                for(int i = 1; i < raw.length; i++) {
                    scores.add(Integer.valueOf(raw[i]));
                }
                ColumnConfig config = columnConfigList.get(columnNum);
                config.setBinAvgScore(scores);
            }
        }

        // release
        closeScanners(scanners);

        return columnConfigList;
    }

}
