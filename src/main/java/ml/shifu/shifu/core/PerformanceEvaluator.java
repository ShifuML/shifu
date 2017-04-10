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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import ml.shifu.shifu.container.ConfusionMatrixObject;
import ml.shifu.shifu.container.PerformanceObject;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.PerformanceResult;
import ml.shifu.shifu.core.eval.AreaUnderCurve;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.JSONUtils;

import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * PerformanceEvaluator class is to evaluate the performance of model. If the
 * evaluation data contains the target column, the PR curve will be generated.
 */
public class PerformanceEvaluator {

    private static Logger log = LoggerFactory.getLogger(PerformanceEvaluator.class);

    private ModelConfig modelConfig;
    private EvalConfig evalConfig;

    public PerformanceEvaluator(ModelConfig modelConfig, EvalConfig evalConfig) {
        this.modelConfig = modelConfig;
        this.evalConfig = evalConfig;
    }

    public void review() throws IOException {
        PathFinder pathFinder = new PathFinder(modelConfig);

        log.info("Loading confusion matrix in {}",
                pathFinder.getEvalMatrixPath(evalConfig, evalConfig.getDataSet().getSource()));

        BufferedReader reader = ShifuFileUtils.getReader(pathFinder.getEvalMatrixPath(evalConfig, evalConfig
                .getDataSet().getSource()), evalConfig.getDataSet().getSource());

        String line = null;

        List<ConfusionMatrixObject> matrixList = new ArrayList<ConfusionMatrixObject>();

        int cnt = 0;

        while((line = reader.readLine()) != null) {
            cnt++;

            String[] raw = line.split("\\|");

            ConfusionMatrixObject matrix = new ConfusionMatrixObject();
            matrix.setTp(Double.parseDouble(raw[0]));
            matrix.setFp(Double.parseDouble(raw[1]));
            matrix.setFn(Double.parseDouble(raw[2]));
            matrix.setTn(Double.parseDouble(raw[3]));

            matrix.setWeightedTp(Double.parseDouble(raw[4]));
            matrix.setWeightedFp(Double.parseDouble(raw[5]));
            matrix.setWeightedFn(Double.parseDouble(raw[6]));
            matrix.setWeightedTn(Double.parseDouble(raw[7]));
            matrix.setScore(Double.parseDouble(raw[8]));

            matrixList.add(matrix);
        }

        if(0 == cnt) {
            log.info("No result read, please check EvalConfusionMatrix file");
            throw new ShifuException(ShifuErrorCode.ERROR_EVALCONFMTR);
        }

        reader.close();

        review(matrixList, cnt);
    }

    public void review(long records) throws IOException {
        if(0 == records) {
            log.info("No result read, please check EvalConfusionMatrix file");
            throw new ShifuException(ShifuErrorCode.ERROR_EVALCONFMTR);
        }

        PathFinder pathFinder = new PathFinder(modelConfig);

        log.info("Loading confusion matrix in {}",
                pathFinder.getEvalMatrixPath(evalConfig, evalConfig.getDataSet().getSource()));

        BufferedReader reader = null;
        try {
            reader = ShifuFileUtils.getReader(pathFinder.getEvalMatrixPath(evalConfig, evalConfig.getDataSet()
                    .getSource()), evalConfig.getDataSet().getSource());
            review(new CMOIterable(reader), records);
        } finally {
            IOUtils.closeQuietly(reader);
        }
    }

    private static class CMOIterable implements Iterable<ConfusionMatrixObject> {

        private BufferedReader reader = null;

        public CMOIterable(BufferedReader reader) {
            if(reader == null) {
                throw new NullPointerException("reader is null");
            }
            this.reader = reader;
        }

        @Override
        public Iterator<ConfusionMatrixObject> iterator() {
            return new Iterator<ConfusionMatrixObject>() {

                private String line;

                @Override
                public boolean hasNext() {
                    try {
                        this.line = CMOIterable.this.reader.readLine();
                        if(this.line == null) {
                            return false;
                        } else {
                            return true;
                        }
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                }

                @Override
                public ConfusionMatrixObject next() {
                    String[] raw = line.split("\\|");

                    ConfusionMatrixObject matrix = new ConfusionMatrixObject();
                    matrix.setTp(Double.parseDouble(raw[0]));
                    matrix.setFp(Double.parseDouble(raw[1]));
                    matrix.setFn(Double.parseDouble(raw[2]));
                    matrix.setTn(Double.parseDouble(raw[3]));

                    matrix.setWeightedTp(Double.parseDouble(raw[4]));
                    matrix.setWeightedFp(Double.parseDouble(raw[5]));
                    matrix.setWeightedFn(Double.parseDouble(raw[6]));
                    matrix.setWeightedTn(Double.parseDouble(raw[7]));
                    matrix.setScore(Double.parseDouble(raw[8]));
                    return matrix;
                }

                @Override
                public void remove() {
                    throw new UnsupportedOperationException();
                }
            };
        }
    }

    public void review(Iterable<ConfusionMatrixObject> matrixList, long records) throws IOException {
        PathFinder pathFinder = new PathFinder(modelConfig);

        // bucketing
        PerformanceResult result = bucketing(matrixList, records, evalConfig.getPerformanceBucketNum(), evalConfig
                .getDataSet().getWeightColumnName() != null);

        Writer writer = null;
        try {
            writer = ShifuFileUtils.getWriter(pathFinder.getEvalPerformancePath(evalConfig, evalConfig.getDataSet()
                    .getSource()), evalConfig.getDataSet().getSource());
            JSONUtils.writeValue(writer, result);
        } catch (IOException e) {
            if(writer != null) {
                writer.close();
            }
        }
    }

    static PerformanceObject setPerformanceObject(ConfusionMatrixObject confMatObject) {
        PerformanceObject po = new PerformanceObject();

        po.binLowestScore = confMatObject.getScore();
        po.tp = confMatObject.getTp();
        po.tn = confMatObject.getTn();
        po.fp = confMatObject.getFp();
        po.fn = confMatObject.getFn();

        po.weightedTp = confMatObject.getWeightedTp();
        po.weightedTn = confMatObject.getWeightedTn();
        po.weightedFp = confMatObject.getWeightedFp();
        po.weightedFn = confMatObject.getWeightedFn();

        // Action Rate, TP + FP / Total;
        po.actionRate = (confMatObject.getTp() + confMatObject.getFp()) / confMatObject.getTotal();

        po.weightedActionRate = (confMatObject.getWeightedTp() + confMatObject.getWeightedFp())
                / confMatObject.getWeightedTotal();

        // recall = TP / (TP+FN)
        po.recall = confMatObject.getTp() / (confMatObject.getTp() + confMatObject.getFn());

        po.weightedRecall = confMatObject.getWeightedTp()
                / (confMatObject.getWeightedTp() + confMatObject.getWeightedFn());

        // precision = TP / (TP+FP)
        po.precision = confMatObject.getTp() / (confMatObject.getTp() + confMatObject.getFp());

        po.weightedPrecision = confMatObject.getWeightedTp()
                / (confMatObject.getWeightedTp() + confMatObject.getWeightedFp());

        // FPR, False Positive Rate (fp/(fp+tn))
        po.fpr = confMatObject.getFp() / (confMatObject.getFp() + confMatObject.getTn());

        po.weightedFpr = confMatObject.getWeightedFp()
                / (confMatObject.getWeightedFp() + confMatObject.getWeightedTn());

        // Lift tp / (number_action * (number_postive / all_unit))
        po.liftUnit = confMatObject.getTp()
                / ((confMatObject.getTp() + confMatObject.getFp()) * (confMatObject.getTp() + confMatObject.getFn()) / confMatObject
                        .getTotal());

        po.weightLiftUnit = confMatObject.getWeightedTp()
                / ((confMatObject.getWeightedTp() + confMatObject.getWeightedFp())
                        * (confMatObject.getWeightedTp() + confMatObject.getWeightedFn()) / confMatObject
                            .getWeightedTotal());

        return po;
    }

    public PerformanceResult bucketing(Iterable<ConfusionMatrixObject> results, long records, int numBucket,
            boolean isWeight) {
        List<PerformanceObject> FPRList = new ArrayList<PerformanceObject>(numBucket + 1);
        List<PerformanceObject> catchRateList = new ArrayList<PerformanceObject>(numBucket + 1);
        List<PerformanceObject> gainList = new ArrayList<PerformanceObject>(numBucket + 1);

        List<PerformanceObject> FPRWeightList = new ArrayList<PerformanceObject>(numBucket + 1);
        List<PerformanceObject> catchRateWeightList = new ArrayList<PerformanceObject>(numBucket + 1);
        List<PerformanceObject> gainWeightList = new ArrayList<PerformanceObject>(numBucket + 1);

        int fpBin = 1, tpBin = 1, gainBin = 1, fpWeightBin = 1, tpWeightBin = 1, gainWeightBin = 1;

        double binCapacity = 1.0 / numBucket;

        PerformanceObject po = null;

        boolean isFirst = true;
        int i = 0;
        for(ConfusionMatrixObject object: results) {
            po = setPerformanceObject(object);
            if(isFirst) {
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

                isFirst = false;
            } else {
                if(po.fpr >= fpBin * binCapacity) {
                    po.binNum = fpBin++;
                    FPRList.add(po);
                }

                if(po.recall >= tpBin * binCapacity) {
                    po.binNum = tpBin++;
                    catchRateList.add(po);
                }

                // prevent 99%
                if((double) (i + 1) / records >= gainBin * binCapacity) {
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
            }
            i++;
        }

        logResult(FPRList, "Bucketing False Positive Rate");

        if(isWeight) {
            logResult(FPRWeightList, "Bucketing Weighted False Positive Rate");
        }

        logResult(catchRateList, "Bucketing Catch Rate");

        if(isWeight) {
            logResult(catchRateWeightList, "Bucketing Weighted Catch Rate");
        }

        logResult(gainList, "Bucketing Action rate");

        if(isWeight) {
            logResult(gainWeightList, "Bucketing Weighted action rate");
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
        logAucResult(result, isWeight);

        return result;
    }

    static void logAucResult(PerformanceResult result, boolean isWeight) {
        log.info("AUC value of ROC: {}", result.areaUnderRoc);
        log.info("AUC value of PR: {}", result.areaUnderPr);

        if(isWeight) {
            log.info("AUC value of weighted ROC: {}", result.weightedAreaUnderRoc);
            log.info("AUC value of weighted PR: {}", result.weightedAreaUnderPr);
        }
    }

    static void logResult(List<PerformanceObject> list, String info) {
        DecimalFormat df = new DecimalFormat("#.####");

        String formatString = "%10s %18s %10s %18s %15s %18s %10s %11s %10s";

        log.info("Start print: " + info);

        log.info(String.format(formatString, "ActionRate", "WeightedActionRate", "Recall", "WeightedRecall",
                "Precision", "WeightedPrecision", "FPR", "WeightedFPR", "BinLowestScore"));

        for(PerformanceObject po: list) {
            log.info(String.format(formatString, df.format(po.actionRate), df.format(po.weightedActionRate),
                    df.format(po.recall), df.format(po.weightedRecall), df.format(po.precision),
                    df.format(po.weightedPrecision), df.format(po.fpr), df.format(po.weightedFpr), po.binLowestScore));
        }

    }
}
