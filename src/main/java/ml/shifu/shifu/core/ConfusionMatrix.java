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
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.Writer;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;

import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.shifu.column.NSColumnUtils;
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
import ml.shifu.shifu.util.Environment;
import ml.shifu.shifu.util.HDFSUtils;
import ml.shifu.shifu.util.JSONUtils;

/**
 * Confusion matrix, hold the confusion matrix computing.
 */
public class ConfusionMatrix {

    private static Logger LOG = LoggerFactory.getLogger(ConfusionMatrix.class);

    public static enum EvaluatorMethod {
        MEAN, MAX, MIN, DEFAULT, MEDIAN;
    }

    /**
     * Model config instance.
     */
    private ModelConfig modelConfig;

    /**
     * Current eval config instance.
     */
    private EvalConfig evalConfig;

    /**
     * Index of target column which is used to read target from eval score output
     */
    private int targetColumnIndex = -1;

    /**
     * Index of first score column which is used to read all eval scores
     */
    private int scoreColumnIndex = -1;

    /**
     * Index of weight column which is used to read weight from eval score output
     */
    private int weightColumnIndex = -1;

    /**
     * Index of first score column which is used to read all eval scores, this is for multiple classification
     */
    private int multiClassScore1Index = -1;

    /**
     * Multiple classification, the number of models
     */
    private int multiClassModelCnt;

    /**
     * Number of meta columns
     */
    private int metaColumns;

    /**
     * The path finder instance to find some specific folder in both local and HDFS
     */
    private PathFinder pathFinder;

    /**
     * Decimal format score output
     */
    private static DecimalFormat SCORE_FORMAT = new DecimalFormat("#.######");

    /**
     * Times to scale raw score
     */
    private double scoreScale;

    /**
     * Set including all positive tags.
     */
    private Set<String> posTags;

    /**
     * Set including all negative tags.
     */
    private Set<String> negTags;

    private Object lock;

    public ConfusionMatrix(ModelConfig modelConfig, EvalConfig evalConfig) throws IOException {
        this(modelConfig, evalConfig, new Object());
    }

    public ConfusionMatrix(ModelConfig modelConfig, EvalConfig evalConfig, Object source) throws IOException {
        this.modelConfig = modelConfig;
        this.evalConfig = evalConfig;
        this.pathFinder = new PathFinder(modelConfig);
        this.lock = source;

        String[] evalScoreHeader = getEvalScoreHeader();
        if(ArrayUtils.isEmpty(evalScoreHeader)) {
            throw new ShifuException(ShifuErrorCode.ERROR_EVAL_NO_EVALSCORE_HEADER);
        }

        if(StringUtils.isEmpty(evalConfig.getPerformanceScoreSelector())) {
            throw new ShifuException(ShifuErrorCode.ERROR_EVAL_SELECTOR_EMPTY);
        }

        if(modelConfig.isRegression()) {
            scoreColumnIndex = getColumnIndex(evalScoreHeader,
                    StringUtils.trimToEmpty(evalConfig.getPerformanceScoreSelector()));
            if(scoreColumnIndex < 0) {
                // the score column is not found in the header of EvalScore
                throw new ShifuException(ShifuErrorCode.ERROR_EVAL_SELECTOR_EMPTY);
            }
        }

        targetColumnIndex = getColumnIndex(evalScoreHeader,
                StringUtils.trimToEmpty(modelConfig.getTargetColumnName(evalConfig)));
        if(targetColumnIndex < 0) {
            // the target column is not found in the header of EvalScore
            throw new ShifuException(ShifuErrorCode.ERROR_EVAL_TARGET_NOT_FOUND);
        }

        weightColumnIndex = getColumnIndex(evalScoreHeader,
                StringUtils.trimToEmpty(evalConfig.getDataSet().getWeightColumnName()));

        // only works for multi classification
        multiClassScore1Index = targetColumnIndex + 2; // target, weight, score1, score2, this is hard code
        try {
            multiClassModelCnt = CommonUtils.getBasicModelsCnt(modelConfig, evalConfig, evalConfig.getDataSet()
                    .getSource());
        } catch (FileNotFoundException e) {
            multiClassModelCnt = 0;
        }

        // Number of meta columns
        metaColumns = evalConfig.getAllMetaColumns(modelConfig).size();

        posTags = new HashSet<String>(modelConfig.getPosTags(evalConfig));
        negTags = new HashSet<String>(modelConfig.getNegTags(evalConfig));

        scoreScale = Double.parseDouble(Environment.getProperty(Constants.SHIFU_SCORE_SCALE,
                Integer.toString(Scorer.DEFAULT_SCORE_SCALE)));
    }

    private int getColumnIndex(String[] headerColumns, String column) {
        int columnIndex = -1;
        for(int i = 0; i < headerColumns.length; i++) {
            if(NSColumnUtils.isColumnEqual(headerColumns[i], column)) {
                columnIndex = i;
                break;
            }
        }
        return columnIndex;
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

    public PerformanceResult bufferedComputeConfusionMatrixAndPerformance(long pigPosTags, long pigNegTags,
            double pigPosWeightTags, double pigNegWeightTags, long records, double maxScore, double minScore,
            String scoreDataPath, String evalPerformancePath, boolean isPrint, boolean isGenerateChart,
            boolean isUseMaxMinScore) throws IOException {
        return bufferedComputeConfusionMatrixAndPerformance(pigPosTags, pigNegTags, pigPosWeightTags, pigNegWeightTags,
                records, maxScore, minScore, scoreDataPath, evalPerformancePath, isPrint, isGenerateChart,
                this.targetColumnIndex, this.scoreColumnIndex, this.weightColumnIndex, isUseMaxMinScore);
    }

    private boolean isGBTNeedConvertScore() {
        String gbtStrategy = evalConfig.getGbtScoreConvertStrategy();
        return CommonConstants.GBT_ALG_NAME.equalsIgnoreCase(modelConfig.getAlgorithm())
                && gbtStrategy != null
                && (gbtStrategy.equalsIgnoreCase(Constants.GBT_SCORE_HALF_CUTOFF_CONVETER) || gbtStrategy
                        .equalsIgnoreCase(Constants.GBT_SCORE_MAXMIN_SCALE_CONVETER));
    }

    private boolean isGBTScoreHalfCutoffStreategy() {
        String gbtStrategy = evalConfig.getGbtScoreConvertStrategy();
        return CommonConstants.GBT_ALG_NAME.equalsIgnoreCase(modelConfig.getAlgorithm()) && gbtStrategy != null
                && gbtStrategy.equalsIgnoreCase(Constants.GBT_SCORE_HALF_CUTOFF_CONVETER);
    }

    private boolean isGBTScoreMaxMinScaleStreategy() {
        String gbtStrategy = evalConfig.getGbtScoreConvertStrategy();
        return CommonConstants.GBT_ALG_NAME.equalsIgnoreCase(modelConfig.getAlgorithm()) && gbtStrategy != null
                && gbtStrategy.equalsIgnoreCase(Constants.GBT_SCORE_MAXMIN_SCALE_CONVETER);
    }

    public PerformanceResult bufferedComputeConfusionMatrixAndPerformance(long pigPosTags, long pigNegTags,
            double pigPosWeightTags, double pigNegWeightTags, long records, double maxPScore, double minPScore,
            String scoreDataPath, String evalPerformancePath, boolean isPrint, boolean isGenerateChart,
            int targetColumnIndex, int scoreColumnIndex, int weightColumnIndex, boolean isUseMaxMinScore)
            throws IOException {
        // 1. compute maxScore and minScore in case some cases score are not in [0, 1]
        double maxScore = 1d * scoreScale, minScore = 0d;

        if(isGBTNeedConvertScore()) {
            // if need convert to [0, 1], just keep max score to 1 and min score to 0 without doing anything
        } else {
            if(isUseMaxMinScore) {
                // TODO some cases maxPScore is already scaled, how to fix that issue
                maxScore = maxPScore;
                minScore = minPScore;
            } else {
                // otherwise, keep [0, 1]
            }
        }

        LOG.info("{} Transformed (scale included) max score is {}, transformed min score is {}",
                evalConfig.getGbtScoreConvertStrategy(), maxScore, minScore);

        SourceType sourceType = evalConfig.getDataSet().getSource();
        List<Scanner> scanners = ShifuFileUtils.getDataScanners(scoreDataPath, sourceType);
        LOG.info("Number of score files is {} in eval {}.", scanners.size(), evalConfig.getName());

        int numBucket = evalConfig.getPerformanceBucketNum();
        boolean hasWeight = StringUtils.isNotBlank(evalConfig.getDataSet().getWeightColumnName());
        boolean isDir = ShifuFileUtils.isDir(pathFinder.getEvalScorePath(evalConfig, sourceType), sourceType);

        List<PerformanceObject> FPRList = new ArrayList<PerformanceObject>(numBucket + 1);
        List<PerformanceObject> catchRateList = new ArrayList<PerformanceObject>(numBucket + 1);
        List<PerformanceObject> gainList = new ArrayList<PerformanceObject>(numBucket + 1);
        List<PerformanceObject> modelScoreList = new ArrayList<PerformanceObject>(numBucket + 1);
        List<PerformanceObject> FPRWeightList = new ArrayList<PerformanceObject>(numBucket + 1);
        List<PerformanceObject> catchRateWeightList = new ArrayList<PerformanceObject>(numBucket + 1);
        List<PerformanceObject> gainWeightList = new ArrayList<PerformanceObject>(numBucket + 1);

        double binScore = (maxScore - minScore) * 1d / numBucket, binCapacity = 1.0 / numBucket, scoreBinCount = 0, scoreBinWeigthedCount = 0;
        int fpBin = 1, tpBin = 1, gainBin = 1, fpWeightBin = 1, tpWeightBin = 1, gainWeightBin = 1, modelScoreBin = 1;
        long index = 0, cnt = 0, invalidTargetCnt = 0, invalidWgtCnt = 0;

        ConfusionMatrixObject prevCmo = buildInitalCmo(pigPosTags, pigNegTags, pigPosWeightTags, pigNegWeightTags,
                maxScore);
        PerformanceObject po = buildFirstPO(prevCmo);

        FPRList.add(po);
        catchRateList.add(po);
        gainList.add(po);
        FPRWeightList.add(po);
        catchRateWeightList.add(po);
        gainWeightList.add(po);
        modelScoreList.add(po);

        boolean isGBTScoreHalfCutoffStreategy = isGBTScoreHalfCutoffStreategy();
        boolean isGBTScoreMaxMinScaleStreategy = isGBTScoreMaxMinScaleStreategy();

        for(Scanner scanner: scanners) {
            while(scanner.hasNext()) {
                if((++cnt) % 100000L == 0L) {
                    LOG.info("Loaded {} records.", cnt);
                }
                if((!isDir) && cnt == 1) {
                    // if the evaluation score file is the local file, skip the first line since we add
                    continue;
                }

                // score is separated by default delimiter in our pig output format
                String[] raw = scanner.nextLine().split(Constants.DEFAULT_ESCAPE_DELIMITER);

                // tag check
                String tag = raw[targetColumnIndex];
                if(StringUtils.isBlank(tag) || (!posTags.contains(tag) && !negTags.contains(tag))) {
                    invalidTargetCnt += 1;
                    continue;
                }

                double weight = 1d;
                // if has weight
                if(weightColumnIndex > 0) {
                    try {
                        weight = Double.parseDouble(raw[weightColumnIndex]);
                    } catch (NumberFormatException e) {
                        invalidWgtCnt += 1;
                    }
                    if(weight < 0d) {
                        invalidWgtCnt += 1;
                        weight = 1d;
                    }
                }

                double score = 0.0;
                try {
                    score = Double.parseDouble(raw[scoreColumnIndex]);
                } catch (NumberFormatException e) {
                    // user set the score column wrong ?
                    if(Math.random() < 0.05) {
                        LOG.warn("The score column - {} is not number. Is score column set correctly?",
                                raw[scoreColumnIndex]);
                    }
                    continue;
                }

                scoreBinCount += 1;
                scoreBinWeigthedCount += weight;

                ConfusionMatrixObject cmo = new ConfusionMatrixObject(prevCmo);
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

                if(isGBTScoreHalfCutoffStreategy) {
                    // half cut off means score <0 then set to 0 and then min score is 0, max score is raw max score,
                    // use max min scale to rescale to [0, 1]
                    if(score < 0d) {
                        score = 0d;
                    }
                    score = ((score - 0) * scoreScale) / (maxPScore - 0);
                } else if(isGBTScoreMaxMinScaleStreategy) {
                    // use max min scaler to make score in [0, 1], don't foget to time scoreScale
                    score = ((score - minPScore) * scoreScale) / (maxPScore - minPScore);
                } else {
                    // do nothing, use current score
                }

                cmo.setScore(Double.parseDouble(SCORE_FORMAT.format(score)));

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
                double validRecordCnt = (double) (index + 1);
                if(validRecordCnt / (pigPosTags + pigNegTags) >= gainBin * binCapacity) {
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

                if((object.getWeightedTp() + object.getWeightedFp()) / object.getWeightedTotal() >= gainWeightBin
                        * binCapacity) {
                    po.binNum = gainWeightBin++;
                    gainWeightList.add(po);
                }

                if((maxScore - (modelScoreBin * binScore)) >= score) {
                    po.binNum = modelScoreBin++;
                    po.scoreCount = scoreBinCount;
                    po.scoreWgtCount = scoreBinWeigthedCount;
                    // System.out.println("score count is " + scoreBinCount);
                    // reset to 0 for next bin score cnt stats
                    scoreBinCount = scoreBinWeigthedCount = 0;
                    modelScoreList.add(po);
                }
                index += 1;
                prevCmo = cmo;
            }
            scanner.close();
        }
        LOG.info("Totally loading {} records with invalid target records {} and invalid weight records {} in eval {}.",
                cnt, invalidTargetCnt, invalidWgtCnt, evalConfig.getName());

        PerformanceResult result = buildPerfResult(FPRList, catchRateList, gainList, modelScoreList, FPRWeightList,
                catchRateWeightList, gainWeightList);

        synchronized(this.lock) {
            if(isPrint) {
                PerformanceEvaluator.logResult(FPRList, "Bucketing False Positive Rate");

                if(hasWeight) {
                    PerformanceEvaluator.logResult(FPRWeightList, "Bucketing Weighted False Positive Rate");
                }

                PerformanceEvaluator.logResult(catchRateList, "Bucketing Catch Rate");

                if(hasWeight) {
                    PerformanceEvaluator.logResult(catchRateWeightList, "Bucketing Weighted Catch Rate");
                }

                PerformanceEvaluator.logResult(gainList, "Bucketing Action Rate");

                if(hasWeight) {
                    PerformanceEvaluator.logResult(gainWeightList, "Bucketing Weighted Action Rate");
                }

                PerformanceEvaluator.logAucResult(result, hasWeight);
            }

            writePerResult2File(evalPerformancePath, result);

            if(isGenerateChart) {
                generateChartAndJsonPerfFiles(hasWeight, result);
            }
        }

        if(cnt == 0) {
            LOG.error("No score read, the EvalScore did not genernate or is null file");
            throw new ShifuException(ShifuErrorCode.ERROR_EVALSCORE);
        }
        return result;
    }

    private void writePerResult2File(String evalPerformancePath, PerformanceResult result) {
        Writer writer = null;
        try {
            writer = ShifuFileUtils.getWriter(evalPerformancePath, evalConfig.getDataSet().getSource());
            JSONUtils.writeValue(writer, result);
        } catch (IOException e) {
            LOG.error("error", e);
        } finally {
            IOUtils.closeQuietly(writer);
        }
    }

    private PerformanceResult buildPerfResult(List<PerformanceObject> FPRList, List<PerformanceObject> catchRateList,
            List<PerformanceObject> gainList, List<PerformanceObject> modelScoreList,
            List<PerformanceObject> FPRWeightList, List<PerformanceObject> catchRateWeightList,
            List<PerformanceObject> gainWeightList) {
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
        return result;
    }

    private void generateChartAndJsonPerfFiles(boolean hasWeight, PerformanceResult result) throws IOException {
        GainChart gc = new GainChart();

        String htmlGainChart = pathFinder.getEvalFilePath(evalConfig.getName(), evalConfig.getName()
                + "_gainchart.html", SourceType.LOCAL);
        LOG.info("Gain chart is generated in {}.", htmlGainChart);
        gc.generateHtml(evalConfig, modelConfig, htmlGainChart, result);

        String htmlPrRocChart = pathFinder.getEvalFilePath(evalConfig.getName(), evalConfig.getName() + "_prroc.html",
                SourceType.LOCAL);
        LOG.info("PR&ROC chart is generated in {}.", htmlPrRocChart);
        gc.generateHtml4PrAndRoc(evalConfig, modelConfig, htmlPrRocChart, result);

        String unitGainChartCsv = pathFinder.getEvalFilePath(evalConfig.getName(), evalConfig.getName()
                + "_unit_wise_gainchart.csv", SourceType.LOCAL);
        LOG.info("Unit-wise gain chart data is generated in {}.", unitGainChartCsv);
        gc.generateCsv(evalConfig, modelConfig, unitGainChartCsv, result.gains);

        if(hasWeight) {
            String weightedGainChartCsv = pathFinder.getEvalFilePath(evalConfig.getName(), evalConfig.getName()
                    + "_weighted_gainchart.csv", SourceType.LOCAL);
            LOG.info("Weighted gain chart data is generated in {}.", weightedGainChartCsv);
            gc.generateCsv(evalConfig, modelConfig, weightedGainChartCsv, result.weightedGains);
        }

        String prCsvFile = pathFinder.getEvalFilePath(evalConfig.getName(), evalConfig.getName() + "_unit_wise_pr.csv",
                SourceType.LOCAL);
        LOG.info("Unit-wise pr data is generated in {}.", prCsvFile);
        gc.generateCsv(evalConfig, modelConfig, prCsvFile, result.pr);

        if(hasWeight) {
            String weightedPrCsvFile = pathFinder.getEvalFilePath(evalConfig.getName(), evalConfig.getName()
                    + "_weighted_pr.csv", SourceType.LOCAL);
            LOG.info("Weighted pr data is generated in {}.", weightedPrCsvFile);
            gc.generateCsv(evalConfig, modelConfig, weightedPrCsvFile, result.weightedPr);
        }

        String rocCsvFile = pathFinder.getEvalFilePath(evalConfig.getName(), evalConfig.getName()
                + "_unit_wise_roc.csv", SourceType.LOCAL);
        LOG.info("Unit-wise roc data is generated in {}.", rocCsvFile);
        gc.generateCsv(evalConfig, modelConfig, rocCsvFile, result.roc);

        if(hasWeight) {
            String weightedRocCsvFile = pathFinder.getEvalFilePath(evalConfig.getName(), evalConfig.getName()
                    + "_weighted_roc.csv", SourceType.LOCAL);
            LOG.info("Weighted roc data is generated in {}.", weightedRocCsvFile);
            gc.generateCsv(evalConfig, modelConfig, weightedRocCsvFile, result.weightedRoc);
        }

        String modelScoreGainChartCsv = pathFinder.getEvalFilePath(evalConfig.getName(), evalConfig.getName()
                + "_modelscore_gainchart.csv", SourceType.LOCAL);
        LOG.info("Model score gain chart data is generated in {}.", modelScoreGainChartCsv);
        gc.generateCsv(evalConfig, modelConfig, modelScoreGainChartCsv, result.modelScoreList);
    }

    private PerformanceObject buildFirstPO(ConfusionMatrixObject prevCmo) {
        PerformanceObject po = PerformanceEvaluator.setPerformanceObject(prevCmo);
        // hit rate == NaN
        po.precision = 1.0;
        po.weightedPrecision = 1.0;
        // lift = NaN
        po.liftUnit = 0.0;
        po.weightLiftUnit = 0.0;
        po.binLowestScore = prevCmo.getScore();
        return po;
    }

    private ConfusionMatrixObject buildInitalCmo(long pigPosTags, long pigNegTags, double pigPosWeightTags,
            double pigNegWeightTags, double maxScore) {
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
        return prevCmo;
    }

    public void computeConfusionMatixForMultipleClassification(long records) throws IOException {
        SourceType sourceType = evalConfig.getDataSet().getSource();

        List<Scanner> scanners = ShifuFileUtils.getDataScanners(pathFinder.getEvalScorePath(evalConfig, sourceType),
                sourceType);
        boolean isDir = ShifuFileUtils.isDir(pathFinder.getEvalScorePath(evalConfig, sourceType), sourceType);
        Set<String> tagSet = new HashSet<String>(modelConfig.getFlattenTags(modelConfig.getPosTags(evalConfig),
                modelConfig.getNegTags(evalConfig)));
        List<Set<String>> tags = modelConfig.getSetTags(modelConfig.getPosTags(evalConfig),
                modelConfig.getNegTags(evalConfig));

        int classes = tags.size();
        long cnt = 0, invalidTargetCnt = 0;

        long[][] confusionMatrix = new long[classes][classes];
        for(Scanner scanner: scanners) {
            while(scanner.hasNext()) {
                if((++cnt) % 100000 == 0) {
                    LOG.info("Loaded " + cnt + " records.");
                }
                if(!isDir && cnt == 1) {
                    // if the evaluation score file is the local file, skip the first line since we add header in
                    continue;
                }

                // score is separated by default delimiter in our pig output format
                String[] raw = scanner.nextLine().split(Constants.DEFAULT_ESCAPE_DELIMITER);

                String tag = raw[targetColumnIndex];
                if(StringUtils.isBlank(tag) || !tagSet.contains(tag)) {
                    invalidTargetCnt += 1;
                    continue;
                }

                double[] scores = new double[classes];

                int maxIndex = -1;
                double maxScore = Double.NEGATIVE_INFINITY;

                if(CommonUtils.isTreeModel(modelConfig.getAlgorithm()) && !modelConfig.getTrain().isOneVsAll()) {
                    // for RF native classification
                    double[] tagCounts = new double[tags.size()];
                    for(int i = this.multiClassScore1Index; i < (raw.length - this.metaColumns); i++) {
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
                } else if((CommonUtils.isTreeModel(modelConfig.getAlgorithm()) || NNConstants.NN_ALG_NAME
                        .equalsIgnoreCase(modelConfig.getAlgorithm())) && modelConfig.getTrain().isOneVsAll()) {
                    // for RF, GBT & NN OneVsAll classification
                    for(int i = this.multiClassScore1Index; i < (classes + this.multiClassScore1Index); i++) {
                        double dd = NumberFormatUtils.getDouble(raw[i], 0d);
                        if(dd > maxScore) {
                            maxScore = dd;
                            maxIndex = i - this.multiClassScore1Index;
                        }
                    }
                } else {
                    // only for NN & Native Multiple classification
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

        LOG.info("Totally loading {} records with invalid target records {} in eval {}.", cnt, invalidTargetCnt,
                evalConfig.getName());

        writeToConfMatrixFile(tags, confusionMatrix);

        // print conf matrix
        LOG.info("Multiple classification confustion matrix:");
        LOG.info(String.format("%15s: %20s", "     ", tags.toString()));
        for(int i = 0; i < confusionMatrix.length; i++) {
            LOG.info(String.format("%15s: %20s", tags.get(i), Arrays.toString(confusionMatrix[i])));
        }
    }

    @SuppressWarnings("deprecation")
    private void writeToConfMatrixFile(List<Set<String>> tags, long[][] confusionMatrix) throws IOException {
        Path localEvalMatrixFile = new Path(pathFinder.getEvalLocalMultiMatrixFile(evalConfig.getName()));

        LOG.info("Saving matrix to local file system: {}.", localEvalMatrixFile);
        if(HDFSUtils.getLocalFS().exists(localEvalMatrixFile)) {
            HDFSUtils.getLocalFS().delete(localEvalMatrixFile);
        }

        BufferedWriter writer = null;
        try {
            writer = ShifuFileUtils.getWriter(localEvalMatrixFile.toString(), SourceType.LOCAL);
            writer.write("\t," + StringUtils.join(tags, ",") + "\n");
            for(int i = 0; i < confusionMatrix.length; i++) {
                StringBuilder sb = new StringBuilder(300);
                sb.append(tags.get(i));
                for(int j = 0; j < confusionMatrix[i].length; j++) {
                    sb.append(",").append(confusionMatrix[i][j]);
                }
                sb.append("\n");
                writer.write(sb.toString());
            }
        } finally {
            writer.close();
        }
    }

    public void computeConfusionMatrix() throws IOException {
        SourceType sourceType = evalConfig.getDataSet().getSource();
        List<Scanner> scanners = ShifuFileUtils.getDataScanners(pathFinder.getEvalScorePath(evalConfig, sourceType),
                sourceType);

        List<ModelResultObject> moList = new ArrayList<ModelResultObject>();

        boolean isDir = ShifuFileUtils.isDir(pathFinder.getEvalScorePath(evalConfig, sourceType), sourceType);

        LOG.info("The size of scanner is {}", scanners.size());

        int cnt = 0;
        for(Scanner scanner: scanners) {
            while(scanner.hasNext()) {
                if((++cnt) % 10000 == 0) {
                    LOG.info("Loaded " + cnt + " records.");
                }

                String[] raw = scanner.nextLine().split("\\|");
                if((!isDir) && cnt == 1) {
                    // if the evaluation score file is the local file, skip the
                    // first line since we add
                    continue;
                }

                String tag = CommonUtils.trimTag(raw[targetColumnIndex]);
                if(StringUtils.isBlank(tag)) {
                    if(Math.random() < 0.01) {
                        LOG.warn("Empty target value!!");
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
                    if(Math.random() < 0.05) {
                        LOG.warn("The score column - {} is not integer. Is score column set correctly?",
                                raw[scoreColumnIndex]);
                    }
                    continue;
                }

                moList.add(new ModelResultObject(score, tag, weight));
            }
            // release resource
            scanner.close();
        }
        LOG.info("Totally loaded " + cnt + " records.");

        if(cnt == 0 || moList.size() == 0) {
            LOG.error("No score read, the EvalScore did not genernate or is null file");
            throw new ShifuException(ShifuErrorCode.ERROR_EVALSCORE);
        }

        ConfusionMatrixCalculator calculator = new ConfusionMatrixCalculator(modelConfig.getPosTags(evalConfig),
                modelConfig.getNegTags(evalConfig), moList);

        BufferedWriter confMatWriter = ShifuFileUtils.getWriter(pathFinder.getEvalMatrixPath(evalConfig, evalConfig
                .getDataSet().getSource()), evalConfig.getDataSet().getSource());
        calculator.calculate(confMatWriter);

        confMatWriter.close();
    }
}
