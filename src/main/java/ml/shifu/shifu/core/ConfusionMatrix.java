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
package ml.shifu.shifu.core;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import ml.shifu.shifu.container.ConfusionMatrixObject;
import ml.shifu.shifu.container.ModelResultObject;
import ml.shifu.shifu.container.PerformanceObject;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.PerformanceResult;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.evaluation.AreaUnderCurve;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.JSONUtils;

import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Confusion matrix, hold the confusion matrix computing
 */
public class ConfusionMatrix {
    public static final Random rd = new Random(System.currentTimeMillis());

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
        if(ArrayUtils.isEmpty(evalScoreHeader)) {
            // no EvalScore header is detected
            throw new ShifuException(ShifuErrorCode.ERROR_EVAL_NO_EVALSCORE_HEADER);
        }

        if(StringUtils.isEmpty(evalConfig.getPerformanceScoreSelector())) {
            throw new ShifuException(ShifuErrorCode.ERROR_EVAL_SELECTOR_EMPTY);
        }

        scoreColumnIndex = ArrayUtils.indexOf(evalScoreHeader, evalConfig.getPerformanceScoreSelector().trim());
        if(scoreColumnIndex < 0) {
            // the score column is not found in the header of EvalScore
            throw new ShifuException(ShifuErrorCode.ERROR_EVAL_SELECTOR_EMPTY);
        }

        targetColumnIndex = ArrayUtils.indexOf(evalScoreHeader, modelConfig.getTargetColumnName(evalConfig));
        if(targetColumnIndex < 0) {
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
        if(isDir) {
            // find the .pig_header file
            pathHeader = pathFinder.getEvalScoreHeaderPath(evalConfig, sourceType);
        } else {
            // evaluation data file
            pathHeader = pathFinder.getEvalScorePath(evalConfig, sourceType);
        }

        return CommonUtils.getHeaders(pathHeader, "|", sourceType, false);
    }
    
    public void bufferedComputeConfusionMatrixAndPerformance(long pigPosTags,
                                                             long pigNegTags,
                                                             double pigPosWeightTags,
                                                             double pigNegWeightTags,
                                                             long records) throws IOException {
        PathFinder pathFinder = new PathFinder(modelConfig);

        SourceType sourceType = evalConfig.getDataSet().getSource();

        List<Scanner> scanners = ShifuFileUtils.getDataScanners(pathFinder.getEvalScorePath(evalConfig, sourceType), sourceType);

        int numBucket = evalConfig.getPerformanceBucketNum();
        boolean isWeight = evalConfig.getDataSet().getWeightColumnName() != null;
        boolean isDir = ShifuFileUtils.isDir(pathFinder.getEvalScorePath(evalConfig, sourceType), sourceType);
        List<PerformanceObject> FPRList = new ArrayList<PerformanceObject>(numBucket + 1);
        List<PerformanceObject> catchRateList = new ArrayList<PerformanceObject>(numBucket + 1);
        List<PerformanceObject> gainList = new ArrayList<PerformanceObject>(numBucket + 1);

        List<PerformanceObject> FPRWeightList = new ArrayList<PerformanceObject>(numBucket + 1);
        List<PerformanceObject> catchRateWeightList = new ArrayList<PerformanceObject>(numBucket + 1);
        List<PerformanceObject> gainWeightList = new ArrayList<PerformanceObject>(numBucket + 1);

        int fpBin = 1, tpBin = 1, gainBin = 1, fpWeightBin = 1, tpWeightBin = 1, gainWeightBin = 1;
        double binCapacity = 1.0 / numBucket;
        PerformanceObject po = null;
        int i = 0;
        log.info("The size of scanner is {}", scanners.size());

        int cnt = 0;
        List<String> posTags = modelConfig.getPosTags(evalConfig);

        ConfusionMatrixObject prevCmo = new ConfusionMatrixObject();
        prevCmo.setTp(0.0);
        prevCmo.setFp(0.0);
        prevCmo.setFn(pigPosTags);
        prevCmo.setTn(pigNegTags);
        prevCmo.setWeightedTp(0.0);
        prevCmo.setWeightedFp(0.0);
        prevCmo.setWeightedFn(pigPosWeightTags);
        prevCmo.setWeightedTn(pigNegWeightTags);
        prevCmo.setScore(1000);

        po = PerformanceEvaluator.setPerformanceObject(prevCmo);
        // hit rate == NaN
        po.precision = 1.0;
        po.weightedPrecision = 1.0;

        // lift = NaN
        po.liftUnit = 0.0;
        po.weightLiftUnit = 0.0;

        FPRList.add(po);
        catchRateList.add(po);
        gainList.add(po);
        FPRWeightList.add(po);
        catchRateWeightList.add(po);
        gainWeightList.add(po);
        for (Scanner scanner : scanners) {
            while (scanner.hasNext()) {
                if ((++cnt) % 100000 == 0) {
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
                double score = 0.0;
                try {
                    score = Double.parseDouble(raw[scoreColumnIndex]);
                } catch (NumberFormatException e) {
                    // user set the score column wrong ?
                    if (rd.nextDouble() < 0.05) {
                        log.warn("The score column - {} is not integer. Is score column set correctly?", raw[scoreColumnIndex]);
                    }
                    continue;
                }

                ConfusionMatrixObject cmo = new ConfusionMatrixObject(prevCmo);

                // TODO enable scaling factor
                if (posTags.contains(tag)) {
                    // Positive Instance
                    cmo.setTp(cmo.getTp() + 1);
                    cmo.setFn(cmo.getFn() - 1);
                    cmo.setWeightedTp(cmo.getWeightedTp() + weight * 1.0);
                    cmo.setWeightedFn(cmo.getWeightedFn() - weight * 1.0);
                } else {
                    // Negative Instance
                    cmo.setFp(cmo.getFp() + 1);
                    cmo.setTn(cmo.getTn() - 1);
                    cmo.setWeightedFp(cmo.getWeightedFp() + weight * 1.0);
                    cmo.setWeightedTn(cmo.getWeightedTn() - weight * 1.0);
                }

                cmo.setScore(score);

                ConfusionMatrixObject object = cmo;
                po = PerformanceEvaluator.setPerformanceObject(object);
                if (po.fpr >= fpBin * binCapacity) {
                    po.binNum = fpBin++;
                    FPRList.add(po);
                }

                if (po.recall >= tpBin * binCapacity) {
                    po.binNum = tpBin++;
                    catchRateList.add(po);
                }

                // prevent 99%
                if ((double) (i + 1) / records >= gainBin * binCapacity) {
                    po.binNum = gainBin++;
                    gainList.add(po);
                }

                if (po.weightedFpr >= fpWeightBin * binCapacity) {
                    po.binNum = fpWeightBin++;
                    FPRWeightList.add(po);
                }

                if (po.weightedRecall >= tpWeightBin * binCapacity) {
                    po.binNum = tpWeightBin++;
                    catchRateWeightList.add(po);
                }

                if ((object.getWeightedTp() + object.getWeightedFp() + 1) / object.getWeightedTotal() >= gainWeightBin * binCapacity) {
                    po.binNum = gainWeightBin++;
                    gainWeightList.add(po);

                }
                i++;
                prevCmo = cmo;
            }
            scanner.close();
        }
        log.info("Totally loaded " + cnt + " records.");

        PerformanceEvaluator.logResult(FPRList, "Bucketing False Positive Rate");

        if (isWeight) {
            PerformanceEvaluator.logResult(FPRWeightList, "Bucketing Weighted False Positive Rate");
        }

        PerformanceEvaluator.logResult(catchRateList, "Bucketing Catch Rate");

        if (isWeight) {
            PerformanceEvaluator.logResult(catchRateWeightList, "Bucketing Weighted Catch Rate");
        }

        PerformanceEvaluator.logResult(gainList, "Bucketing Action rate");

        if (isWeight) {
            PerformanceEvaluator.logResult(gainWeightList, "Bucketing Weighted action rate");
        }

        PerformanceResult result = new PerformanceResult();

        result.version = Constants.version;
        result.pr = catchRateList;
        result.weightedPr = catchRateWeightList;
        result.roc = FPRList;
        result.weightedRoc = FPRWeightList;
        result.gains = gainList;
        result.weightedGains = gainWeightList;

        // Calculate area under curve
        result.areaUnderRoc = AreaUnderCurve.ofRoc(result.roc);
        result.weightedAreaUnderRoc = AreaUnderCurve.ofWeightedRoc(result.weightedRoc);
        result.areaUnderPr = AreaUnderCurve.ofPr(result.pr);
        result.weightedAreaUnderPr = AreaUnderCurve.ofWeightedPr(result.weightedPr);
        
        Writer writer = null;
        try {
            writer = ShifuFileUtils.getWriter(pathFinder.getEvalPerformancePath(evalConfig, evalConfig.getDataSet().getSource()), evalConfig
                    .getDataSet().getSource());
            JSONUtils.writeValue(writer, result);
        } catch (IOException e) {
            IOUtils.closeQuietly(writer);
        }
        if (cnt == 0) {
            log.error("No score read, the EvalScore did not genernate or is null file");
            throw new ShifuException(ShifuErrorCode.ERROR_EVALSCORE);
        }

    }

    public void bufferedComputeConfusionMatrix(long pigPosTags, long pigNegTags, double pigPosWeightTags,
            double pigNegWeightTags) throws IOException {
        PathFinder pathFinder = new PathFinder(modelConfig);

        SourceType sourceType = evalConfig.getDataSet().getSource();

        List<Scanner> scanners = ShifuFileUtils.getDataScanners(pathFinder.getEvalScorePath(evalConfig, sourceType),
                sourceType);

        boolean isDir = ShifuFileUtils.isDir(pathFinder.getEvalScorePath(evalConfig, sourceType), sourceType);

        log.info("The size of scanner is {}", scanners.size());

        int cnt = 0;

        List<String> posTags = modelConfig.getPosTags(evalConfig);

        BufferedWriter confMatWriter = ShifuFileUtils.getWriter(pathFinder.getEvalMatrixPath(evalConfig, evalConfig
                .getDataSet().getSource()), evalConfig.getDataSet().getSource());

        ConfusionMatrixObject prevCmo = new ConfusionMatrixObject();
        prevCmo.setTp(0.0);
        prevCmo.setFp(0.0);
        prevCmo.setFn(pigPosTags);
        prevCmo.setTn(pigNegTags);
        prevCmo.setWeightedTp(0.0);
        prevCmo.setWeightedFp(0.0);
        prevCmo.setWeightedFn(pigPosWeightTags);
        prevCmo.setWeightedTn(pigNegWeightTags);
        prevCmo.setScore(1000);

        ConfusionMatrixCalculator.saveConfusionMaxtrixWithWriter(confMatWriter, prevCmo);

        for (Scanner scanner : scanners) {
            while(scanner.hasNext()) {
                if((++cnt) % 100000 == 0) {
                    log.info("Loaded " + cnt + " records.");
                }

                String[] raw = scanner.nextLine().split("\\|");

                if((!isDir) && cnt == 1) {
                    // if the evaluation score file is the local file, skip the
                    // first line since we add
                    continue;
                }

                String tag = raw[targetColumnIndex];
                if(StringUtils.isBlank(tag)) {
                    if(rd.nextDouble() < 0.01) {
                        log.warn("Empty target value!!");
                    }

                    continue;
                }

                double weight = 1.0d;
                if(this.weightColumnIndex > 0) {
                    try {
                        weight = Double.parseDouble(raw[1]);
                    } catch (NumberFormatException e) {
                        // Do nothing
                    }
                }

                double score = 0.0;
                try {
                    score = Double.parseDouble(raw[scoreColumnIndex]);
                } catch (NumberFormatException e) {
                    // user set the score column wrong ?
                    if(rd.nextDouble() < 0.05) {
                        log.warn("The score column - {} is not integer. Is score column set correctly?",
                                raw[scoreColumnIndex]);
                    }
                    continue;
                }

                ConfusionMatrixObject cmo = new ConfusionMatrixObject(prevCmo);
                // TODO enable scaling factor
                if(posTags.contains(tag)) {
                    // Positive Instance
                    cmo.setTp(cmo.getTp() + 1);
                    cmo.setFn(cmo.getFn() - 1);
                    cmo.setWeightedTp(cmo.getWeightedTp() + weight * 1.0);
                    cmo.setWeightedFn(cmo.getWeightedFn() - weight * 1.0);
                } else {
                    // Negative Instance
                    cmo.setFp(cmo.getFp() + 1);
                    cmo.setTn(cmo.getTn() - 1);
                    cmo.setWeightedFp(cmo.getWeightedFp() + weight * 1.0);
                    cmo.setWeightedTn(cmo.getWeightedTn() - weight * 1.0);
                }

                cmo.setScore(score);
                ConfusionMatrixCalculator.saveConfusionMaxtrixWithWriter(confMatWriter, cmo);
                prevCmo = cmo;
            }
            scanner.close();
        }

        log.info("Totally loaded " + cnt + " records.");

        if(cnt == 0) {
            log.error("No score read, the EvalScore did not genernate or is null file");
            throw new ShifuException(ShifuErrorCode.ERROR_EVALSCORE);
        }
        confMatWriter.close();
    }
    
    public void computeConfusionMatrix() throws IOException {

        PathFinder pathFinder = new PathFinder(modelConfig);

        SourceType sourceType = evalConfig.getDataSet().getSource();

        List<Scanner> scanners = ShifuFileUtils.getDataScanners(pathFinder.getEvalScorePath(evalConfig, sourceType),
                sourceType);

        List<ModelResultObject> moList = new ArrayList<ModelResultObject>();

        boolean isDir = ShifuFileUtils.isDir(pathFinder.getEvalScorePath(evalConfig, sourceType), sourceType);

        log.info("The size of scanner is {}", scanners.size());

        int cnt = 0;
        for(Scanner scanner: scanners) {
            while(scanner.hasNext()) {
                if((++cnt) % 10000 == 0) {
                    log.info("Loaded " + cnt + " records.");
                }

                String[] raw = scanner.nextLine().split("\\|");
                if((!isDir) && cnt == 1) {
                    // if the evaluation score file is the local file, skip the
                    // first line since we add
                    continue;
                }

                String tag = raw[targetColumnIndex];
                if(StringUtils.isBlank(tag)) {
                    if(rd.nextDouble() < 0.01) {
                        log.warn("Empty target value!!");
                    }

                    continue;
                }

                double weight = 1.0d;
                if(this.weightColumnIndex > 0) {
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
                    if(rd.nextDouble() < 0.05) {
                        log.warn("The score column - {} is not integer. Is score column set correctly?",
                                raw[scoreColumnIndex]);
                    }
                    continue;
                }

                moList.add(new ModelResultObject(score, tag, weight));
            }

            // release resource
            scanner.close();
        }

        log.info("Totally loaded " + cnt + " records.");

        if(cnt == 0 || moList.size() == 0) {
            log.error("No score read, the EvalScore did not genernate or is null file");
            throw new ShifuException(ShifuErrorCode.ERROR_EVALSCORE);
        }

        ConfusionMatrixCalculator calculator = new ConfusionMatrixCalculator(modelConfig.getPosTags(evalConfig),
                modelConfig.getNegTags(evalConfig), moList);

        PathFinder finder = new PathFinder(modelConfig);

        BufferedWriter confMatWriter = ShifuFileUtils.getWriter(finder.getEvalMatrixPath(evalConfig, evalConfig
                .getDataSet().getSource()), evalConfig.getDataSet().getSource());

        calculator.calculate(confMatWriter);

        confMatWriter.close();

    }
}
