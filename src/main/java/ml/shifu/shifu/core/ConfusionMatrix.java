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
package ml.shifu.shifu.core;

import ml.shifu.shifu.container.ModelResultObject;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

/**
 * Confusion matrix, hold the confusion matrix computing
 */
public class ConfusionMatrix {
    public static Random rd = new Random(System.currentTimeMillis());

    enum EvaluatorMethod {
        MEAN, MAX, MIN, DEFAULT, MEDIAN;
    }

    private static Logger log = LoggerFactory.getLogger(ConfusionMatrix.class);

    private ModelConfig modelConfig;
    private EvalConfig evalConfig;

    private int targetColumnIndex = -1;
    private int scoreColumnIndex = -1;
    private int weightColumnIndex = -1;


    public ConfusionMatrix(ModelConfig modelConfig, EvalConfig evalConfig) throws IOException {
        this.modelConfig = modelConfig;
        this.evalConfig = evalConfig;

        String[] evalScoreHeader = getEvalScoreHeader();
        if (ArrayUtils.isEmpty(evalScoreHeader)) {
            // no EvalScore header is detected
            throw new ShifuException(ShifuErrorCode.ERROR_EVAL_NO_EVALSCORE_HEADER);
        }

        if (StringUtils.isEmpty(evalConfig.getPerformanceScoreSelector())) {
            throw new ShifuException(ShifuErrorCode.ERROR_EVAL_SELECTOR_EMPTY);
        }

        scoreColumnIndex = ArrayUtils.indexOf(evalScoreHeader, evalConfig.getPerformanceScoreSelector().trim());
        if (scoreColumnIndex < 0) {
            // the score column is not found in the header of EvalScore
            throw new ShifuException(ShifuErrorCode.ERROR_EVAL_SELECTOR_EMPTY);
        }

        targetColumnIndex = ArrayUtils.indexOf(evalScoreHeader, modelConfig.getTargetColumnName(evalConfig));
        if (targetColumnIndex < 0) {
            // the target column is not found in the header of EvalScore
            throw new ShifuException(ShifuErrorCode.ERROR_EVAL_TARGET_NOT_FOUND);
        }

        weightColumnIndex = ArrayUtils.indexOf(evalScoreHeader, evalConfig.getDataSet().getWeightColumnName());
    }

    /**
     * @return
     * @throws IOException
     */
    private String[] getEvalScoreHeader() throws IOException {
        PathFinder pathFinder = new PathFinder(modelConfig);
        SourceType sourceType = evalConfig.getDataSet().getSource();

        String pathHeader = null;
        boolean isDir = ShifuFileUtils.isDir(pathFinder.getEvalScorePath(evalConfig, sourceType), sourceType);
        if (isDir) {
            // find the .pig_header file
            pathHeader = pathFinder.getEvalScoreHeaderPath(evalConfig, sourceType);
        } else {
            // evaluation data file
            pathHeader = pathFinder.getEvalScorePath(evalConfig, sourceType);
        }

        return CommonUtils.getHeaders(pathHeader, "|", sourceType);
    }


    public void computeConfusionMatrix() throws IOException {

        PathFinder pathFinder = new PathFinder(modelConfig);

        SourceType sourceType = evalConfig.getDataSet().getSource();

        List<Scanner> scanners = ShifuFileUtils.getDataScanners(pathFinder.getEvalScorePath(evalConfig, sourceType), sourceType);

        List<ModelResultObject> moList = new ArrayList<ModelResultObject>();

        boolean isDir = ShifuFileUtils.isDir(pathFinder.getEvalScorePath(evalConfig, sourceType), sourceType);

        log.info("The size of scanner is {}", scanners.size());

        int cnt = 0;
        for (Scanner scanner : scanners) {
            while (scanner.hasNext()) {
                if ((++cnt) % 10000 == 0) {
                    log.info("Loaded " + cnt + " records.");
                }

                String[] raw = scanner.nextLine().split("\\|");
                if ((!isDir) && cnt == 1) {
                    // if the evaluation score file is the local file, skip the
                    // first line since we add
                    continue;
                }

                String tag = raw[targetColumnIndex];
                if (StringUtils.isBlank(tag)) {
                    if (rd.nextDouble() < 0.01) {
                        log.warn("Empty target value!!");
                    }

                    continue;
                }

                double weight = 1.0d;
                if (this.weightColumnIndex > 0) {
                    try {
                        weight = Double.parseDouble(raw[1]);
                    } catch (NumberFormatException e) {
                        // Do nothing
                    }
                }

                double score = 0;
                try {
                    score = Double.parseDouble(raw[scoreColumnIndex]);
                } catch (NumberFormatException e) {
                    // user set the score column wrong ?
                    if (rd.nextDouble() < 0.05) {
                        log.warn("The score column - {} is not integer. Is score column set correctly?", raw[scoreColumnIndex]);
                    }
                    continue;
                }

                moList.add(new ModelResultObject(score, tag, weight));
            }

            // release resource
            scanner.close();
        }

        log.info("Totally loaded " + cnt + " records.");

        if (cnt == 0 || moList.size() == 0) {
            log.error("No score read, the EvalScore did not genernate or is null file");
            throw new ShifuException(ShifuErrorCode.ERROR_EVALSCORE);
        }

        ConfusionMatrixCalculator calculator = new ConfusionMatrixCalculator(
                modelConfig.getPosTags(evalConfig), modelConfig.getNegTags(evalConfig), moList);


        PathFinder finder = new PathFinder(modelConfig);

        BufferedWriter confMatWriter = ShifuFileUtils.getWriter(
                finder.getEvalMatrixPath(evalConfig, evalConfig.getDataSet().getSource()),
                evalConfig.getDataSet().getSource());

        calculator.calculate(confMatWriter);

        confMatWriter.close();

    }
}
