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
package ml.shifu.shifu.core;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;

import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.shifu.container.ConfusionMatrixObject;
import ml.shifu.shifu.container.ModelResultObject;
import ml.shifu.shifu.container.PerformanceObject;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.PerformanceResult;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.nn.NNConstants;
import ml.shifu.shifu.core.eval.AreaUnderCurve;
import ml.shifu.shifu.core.eval.GainChart;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.HDFSUtils;
import ml.shifu.shifu.util.JSONUtils;

import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Confusion matrix, hold the confusion matrix computing
 * TODO refactor to binary and multiple classification
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

    private int multiClassScore1Index = -1;

    private int multiClassModelCnt;

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

        if(modelConfig.isRegression()) {
            scoreColumnIndex = ArrayUtils.indexOf(evalScoreHeader, evalConfig.getPerformanceScoreSelector().trim());
            if(scoreColumnIndex < 0) {
                // the score column is not found in the header of EvalScore
                throw new ShifuException(ShifuErrorCode.ERROR_EVAL_SELECTOR_EMPTY);
            }
        }

        targetColumnIndex = ArrayUtils.indexOf(evalScoreHeader, modelConfig.getTargetColumnName(evalConfig));
        if(targetColumnIndex < 0) {
            // the target column is not found in the header of EvalScore
            throw new ShifuException(ShifuErrorCode.ERROR_EVAL_TARGET_NOT_FOUND);
        }

        weightColumnIndex = ArrayUtils.indexOf(evalScoreHeader, evalConfig.getDataSet().getWeightColumnName());

        // only works for multi classfication
        multiClassScore1Index = targetColumnIndex + 2; // taget, weight, score1, score2
        multiClassModelCnt = (evalScoreHeader.length - multiClassScore1Index) / modelConfig.getTags().size();
    }

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

    public void bufferedComputeConfusionMatrixAndPerformance(long pigPosTags, long pigNegTags, double pigPosWeightTags,
            double pigNegWeightTags, long records, int maxScore, int minScore) throws IOException {
        log.info("Max score is {}, min score is {}", maxScore, minScore);

        PathFinder pathFinder = new PathFinder(modelConfig);

        if(!CommonConstants.GBT_ALG_NAME.equalsIgnoreCase(modelConfig.getTrain().getAlgorithm())) {
            // if not GBT model, NN/LR, are all 0-1000, only for GBT, maxScore and minScore may not be 1000 and 0
            maxScore = 1000;
            minScore = 0;
        }

        boolean gbtConvertToProb = isGBTConvertToProb();
        if(gbtConvertToProb) {
            log.debug(" set max score to 1000,raw  max is {}, raw min is {}", maxScore, minScore);
            maxScore = 1000;
            minScore = 0;
        }

        SourceType sourceType = evalConfig.getDataSet().getSource();

        List<Scanner> scanners = ShifuFileUtils.getDataScanners(pathFinder.getEvalScorePath(evalConfig, sourceType),
                sourceType);

        int numBucket = evalConfig.getPerformanceBucketNum();
        boolean isWeight = evalConfig.getDataSet().getWeightColumnName() != null;
        boolean isDir = ShifuFileUtils.isDir(pathFinder.getEvalScorePath(evalConfig, sourceType), sourceType);
        List<PerformanceObject> FPRList = new ArrayList<PerformanceObject>(numBucket + 1);
        List<PerformanceObject> catchRateList = new ArrayList<PerformanceObject>(numBucket + 1);
        List<PerformanceObject> gainList = new ArrayList<PerformanceObject>(numBucket + 1);
        // bucketing model score
        List<PerformanceObject> modelScoreList = new ArrayList<PerformanceObject>(numBucket + 1);

        List<PerformanceObject> FPRWeightList = new ArrayList<PerformanceObject>(numBucket + 1);
        List<PerformanceObject> catchRateWeightList = new ArrayList<PerformanceObject>(numBucket + 1);
        List<PerformanceObject> gainWeightList = new ArrayList<PerformanceObject>(numBucket + 1);

        double binScore = (maxScore - minScore) * 1d / numBucket;

        int fpBin = 1, tpBin = 1, gainBin = 1, fpWeightBin = 1, tpWeightBin = 1, gainWeightBin = 1, modelScoreBin = 1;
        double binCapacity = 1.0 / numBucket;
        PerformanceObject po = null;
        int i = 0;
        log.info("The size of scanner is {}", scanners.size());

        int cnt = 0;
        Set<String> posTags = new HashSet<String>(modelConfig.getPosTags(evalConfig));
        Set<String> negTags = new HashSet<String>(modelConfig.getNegTags(evalConfig));

        ConfusionMatrixObject prevCmo = new ConfusionMatrixObject();
        prevCmo.setTp(0.0);
        prevCmo.setFp(0.0);
        prevCmo.setFn(pigPosTags);
        prevCmo.setTn(pigNegTags);
        prevCmo.setWeightedTp(0.0);
        prevCmo.setWeightedFp(0.0);
        prevCmo.setWeightedFn(pigPosWeightTags);
        prevCmo.setWeightedTn(pigNegWeightTags);
        prevCmo.setScore(maxScore);

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
        modelScoreList.add(po);
        for(Scanner scanner: scanners) {
            while(scanner.hasNext()) {
                if((++cnt) % 100000 == 0) {
                    log.info("Loaded " + cnt + " records.");
                }

                String[] raw = scanner.nextLine().split("\\|");

                if((!isDir) && cnt == 1) {
                    // if the evaluation score file is the local file, skip the first line since we add
                    continue;
                }

                String tag = raw[targetColumnIndex];
                if(StringUtils.isBlank(tag) || (!posTags.contains(tag) && !negTags.contains(tag))) {
                    if(rd.nextDouble() < 0.01) {
                        log.warn("Empty target value or invalid target value: {}!!", tag);
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
                if(cnt == 1 && CommonConstants.GBT_ALG_NAME.equalsIgnoreCase(modelConfig.getAlgorithm())
                        && !gbtConvertToProb) {
                    // for gbdt, the result maybe not in [0, 1], set first score to make the upper score bould clear
                    po.binLowestScore = score;
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

                ConfusionMatrixObject object = cmo;
                po = PerformanceEvaluator.setPerformanceObject(object);
                if(po.fpr >= fpBin * binCapacity) {
                    po.binNum = fpBin++;
                    FPRList.add(po);
                }

                if(po.recall >= tpBin * binCapacity) {
                    po.binNum = tpBin++;
                    catchRateList.add(po);
                }

                // prevent 99%
                // if((double) (i + 1) / records >= gainBin * binCapacity) {
                double validRecordCnt = (double) (i + 1);
                if ( validRecordCnt / (pigPosTags + pigNegTags) >= gainBin * binCapacity ) {
                    po.binNum = gainBin++;
                    gainList.add(po);
                }

                if(po.weightedFpr >= fpWeightBin * binCapacity) {
                    po.binNum = fpWeightBin++;
                    FPRWeightList.add(po);
                }

                if(po.weightedRecall >= tpWeightBin * binCapacity) {
                    po.binNum = tpWeightBin++;
                    catchRateWeightList.add(po);
                }

                if((object.getWeightedTp() + object.getWeightedFp() + 1) / object.getWeightedTotal() >= gainWeightBin
                        * binCapacity) {
                    po.binNum = gainWeightBin++;
                    gainWeightList.add(po);
                }

                if((maxScore - (int) (modelScoreBin * binScore)) >= score) {
                    po.binNum = modelScoreBin++;
                    modelScoreList.add(po);
                }
                i++;
                prevCmo = cmo;
            }
            scanner.close();
        }
        log.info("Totally loaded " + cnt + " records.");

        PerformanceEvaluator.logResult(FPRList, "Bucketing False Positive Rate");

        if(isWeight) {
            PerformanceEvaluator.logResult(FPRWeightList, "Bucketing Weighted False Positive Rate");
        }

        PerformanceEvaluator.logResult(catchRateList, "Bucketing Catch Rate");

        if(isWeight) {
            PerformanceEvaluator.logResult(catchRateWeightList, "Bucketing Weighted Catch Rate");
        }

        PerformanceEvaluator.logResult(gainList, "Bucketing Action rate");

        if(isWeight) {
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
        result.modelScoreList = modelScoreList;

        // Calculate area under curve
        result.areaUnderRoc = AreaUnderCurve.ofRoc(result.roc);
        result.weightedAreaUnderRoc = AreaUnderCurve.ofWeightedRoc(result.weightedRoc);
        result.areaUnderPr = AreaUnderCurve.ofPr(result.pr);
        result.weightedAreaUnderPr = AreaUnderCurve.ofWeightedPr(result.weightedPr);
        PerformanceEvaluator.logAucResult(result, isWeight);

        Writer writer = null;
        try {
            writer = ShifuFileUtils.getWriter(pathFinder.getEvalPerformancePath(evalConfig, evalConfig.getDataSet()
                    .getSource()), evalConfig.getDataSet().getSource());
            JSONUtils.writeValue(writer, result);
        } catch (IOException e) {
            log.error("error", e);
        } finally {
            IOUtils.closeQuietly(writer);
        }

        String htmlGainChart = pathFinder.getEvalFilePath(evalConfig.getName(), evalConfig.getName()
                + "_gainchart.html", SourceType.LOCAL);
        log.info("Gain chart is generated in {}.", htmlGainChart);
        GainChart gc = new GainChart();
        gc.generateHtml(evalConfig, modelConfig, htmlGainChart, result);

        String unitGainChartCsv = pathFinder.getEvalFilePath(evalConfig.getName(), evalConfig.getName()
                + "_unit_wise_gainchart.csv", SourceType.LOCAL);
        log.info("Unit-wise gain chart data is generated in {}.", unitGainChartCsv);
        gc.generateCsv(evalConfig, modelConfig, unitGainChartCsv, result.gains);

        if(isWeight) {
            String weightedGainChartCsv = pathFinder.getEvalFilePath(evalConfig.getName(), evalConfig.getName()
                    + "_weighted_gainchart.csv", SourceType.LOCAL);
            log.info("Weighted gain chart data is generated in {}.", weightedGainChartCsv);
            gc.generateCsv(evalConfig, modelConfig, weightedGainChartCsv, result.weightedGains);
        }

        String prCsvFile = pathFinder.getEvalFilePath(evalConfig.getName(), evalConfig.getName() + "_unit_wise_pr.csv",
                SourceType.LOCAL);
        log.info("Unit-wise pr data is generated in {}.", prCsvFile);
        gc.generateCsv(evalConfig, modelConfig, prCsvFile, result.pr);

        if(isWeight) {
            String weightedPrCsvFile = pathFinder.getEvalFilePath(evalConfig.getName(), evalConfig.getName()
                    + "_weighted_pr.csv", SourceType.LOCAL);
            log.info("Weighted pr data is generated in {}.", weightedPrCsvFile);
            gc.generateCsv(evalConfig, modelConfig, weightedPrCsvFile, result.weightedPr);
        }

        String rocCsvFile = pathFinder.getEvalFilePath(evalConfig.getName(), evalConfig.getName()
                + "_unit_wise_roc.csv", SourceType.LOCAL);
        log.info("Unit-wise roc data is generated in {}.", rocCsvFile);
        gc.generateCsv(evalConfig, modelConfig, rocCsvFile, result.roc);

        if(isWeight) {
            String weightedRocCsvFile = pathFinder.getEvalFilePath(evalConfig.getName(), evalConfig.getName()
                    + "_weighted_roc.csv", SourceType.LOCAL);
            log.info("Weighted roc data is generated in {}.", weightedRocCsvFile);
            gc.generateCsv(evalConfig, modelConfig, weightedRocCsvFile, result.weightedRoc);
        }

        String modelScoreGainChartCsv = pathFinder.getEvalFilePath(evalConfig.getName(), evalConfig.getName()
                + "_modelscore_gainchart.csv", SourceType.LOCAL);
        log.info("Model score gain chart data is generated in {}.", modelScoreGainChartCsv);
        gc.generateCsv(evalConfig, modelConfig, modelScoreGainChartCsv, result.modelScoreList);

        if(cnt == 0) {
            log.error("No score read, the EvalScore did not genernate or is null file");
            throw new ShifuException(ShifuErrorCode.ERROR_EVALSCORE);
        }
    }

    private boolean isGBTConvertToProb() {
        return CommonConstants.GBT_ALG_NAME.equalsIgnoreCase(modelConfig.getTrain().getAlgorithm())
                && evalConfig.getGbtConvertToProb();
    }

    @SuppressWarnings("deprecation")
    public void computeConfusionMatixForMultipleClassification(long records) throws IOException {
        PathFinder pathFinder = new PathFinder(modelConfig);
        SourceType sourceType = evalConfig.getDataSet().getSource();

        List<Scanner> scanners = ShifuFileUtils.getDataScanners(pathFinder.getEvalScorePath(evalConfig, sourceType),
                sourceType);
        boolean isDir = ShifuFileUtils.isDir(pathFinder.getEvalScorePath(evalConfig, sourceType), sourceType);
        int cnt = 0;
        Set<String> posTags = new HashSet<String>(modelConfig.getPosTags(evalConfig));
        Set<String> negTags = new HashSet<String>(modelConfig.getNegTags(evalConfig));
        Set<String> tagSet = new HashSet<String>(modelConfig.getFlattenTags(modelConfig.getPosTags(evalConfig),
                modelConfig.getNegTags(evalConfig)));
        // List<String> tags = modelConfig.getFlattenTags(modelConfig.getPosTags(evalConfig),
        // modelConfig.getNegTags(evalConfig));
        List<Set<String>> tags = modelConfig.getSetTags(modelConfig.getPosTags(evalConfig),
                modelConfig.getNegTags(evalConfig));

        int classes = tags.size();

        long[][] confusionMatrix = new long[classes][classes];
        for(Scanner scanner: scanners) {
            while(scanner.hasNext()) {
                if((++cnt) % 100000 == 0) {
                    log.info("Loaded " + cnt + " records.");
                }

                String[] raw = scanner.nextLine().split("\\|");

                if(!isDir && cnt == 1) {
                    // if the evaluation score file is the local file, skip the first line since we add
                    continue;
                }

                String tag = raw[targetColumnIndex];
                if(modelConfig.isRegression()) {
                    if(StringUtils.isBlank(tag) || (!posTags.contains(tag) && !negTags.contains(tag))) {
                        if(rd.nextDouble() < 0.01) {
                            log.warn("Empty or invalid target value!!");
                        }
                        continue;
                    }
                } else {
                    if(StringUtils.isBlank(tag) || !tagSet.contains(tag)) {
                        if(rd.nextDouble() < 0.01) {
                            log.warn("Empty or invalid target value!!");
                        }
                        continue;
                    }
                }

                double[] scores = new double[classes];

                int maxIndex = -1;
                double maxScore = Double.NEGATIVE_INFINITY;

                if(CommonUtils.isDesicionTreeAlgorithm(modelConfig.getAlgorithm())
                        && !modelConfig.getTrain().isOneVsAll()) {
                    // for RF classification
                    double[] tagCounts = new double[tags.size()];
                    for(int i = this.multiClassScore1Index; i < raw.length; i++) {
                        double dd = NumberFormatUtils.getDouble(raw[i], 0d);
                        tagCounts[(int) dd] += 1d;
                    }
                    double maxVotes = -1d;
                    for(int i = 0; i < tagCounts.length; i++) {
                        if(tagCounts[i] > maxVotes) {
                            maxIndex = i;
                            maxScore = maxVotes = tagCounts[i];
                        }
                    }
                } else if((CommonUtils.isDesicionTreeAlgorithm(modelConfig.getAlgorithm()) || NNConstants.NN_ALG_NAME
                        .equalsIgnoreCase(modelConfig.getAlgorithm())) && modelConfig.getTrain().isOneVsAll()) {
                    // for RF & NN OneVsAll classification
                    for(int i = this.multiClassScore1Index; i < raw.length; i++) {
                        double dd = NumberFormatUtils.getDouble(raw[i], 0d);
                        if(dd > maxScore) {
                            maxScore = dd;
                            maxIndex = i - this.multiClassScore1Index;
                        }
                    }
                } else {
                    // only for NN
                    // 1,2,3 4,5,6: 1,2,3 is model 0, 4,5,6 is model 1
                    for(int i = 0; i < classes; i++) {
                        for(int j = 0; j < multiClassModelCnt; j++) {
                            double dd = NumberFormatUtils.getDouble(raw[this.multiClassScore1Index + j * classes + i],
                                    0d);
                            scores[i] += dd;
                        }
                        scores[i] /= multiClassModelCnt;
                        if(scores[i] > maxScore) {
                            maxIndex = i;
                            maxScore = scores[i];
                        }
                    }
                }
                int tagIndex = -1;
                for(int i = 0; i < tags.size(); i++) {
                    if(tags.get(i).contains(tag)) {
                        tagIndex = i;
                        break;
                    }
                }
                confusionMatrix[tagIndex][maxIndex] += 1L;
            }
            scanner.close();
        }

        Path localEvalMatrixFile = new Path(pathFinder.getEvalLocalMultiMatrixFile(evalConfig.getName()));
        log.info("Saving matrix to local file system: {}.", localEvalMatrixFile);
        if(HDFSUtils.getLocalFS().exists(localEvalMatrixFile)) {
            HDFSUtils.getLocalFS().delete(localEvalMatrixFile);
        }

        BufferedWriter writer = null;
        try {
            writer = ShifuFileUtils.getWriter(localEvalMatrixFile.toString(), SourceType.LOCAL);
            writer.write("," + StringUtils.join(tags, ",") + "\n");
            for(int i = 0; i < confusionMatrix.length; i++) {
                StringBuilder sb = new StringBuilder(300);
                sb.append(tags.get(i));
                for(int j = 0; j < confusionMatrix[i].length; j++) {
                    sb.append(",").append(confusionMatrix[i][j]);
                }
                sb.append("\n");
                writer.write(tags.get(i) + "," + sb.toString());
            }
        } finally {
            writer.close();
        }

        log.info("Multiple classification confustion matrix:");
        log.info(String.format("%15s: %20s", "     ", tags.toString()));
        for(int i = 0; i < confusionMatrix.length; i++) {
            log.info(String.format("%15s: %20s", tags.get(i), Arrays.toString(confusionMatrix[i])));
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

        for(Scanner scanner: scanners) {
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

                String tag = CommonUtils.trimTag(raw[targetColumnIndex]);
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
