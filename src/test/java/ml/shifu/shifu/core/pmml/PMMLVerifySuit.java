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
package ml.shifu.shifu.core.pmml;

import java.io.File;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.shifu.shifu.ShifuCLI;
import ml.shifu.shifu.container.obj.ModelTrainConf;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractSpecifCreator;
import ml.shifu.shifu.core.pmml.builder.impl.NNSpecifCreator;
import ml.shifu.shifu.core.processor.ExportModelProcessor;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.ClassificationMap;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.ModelEvaluator;
import org.jpmml.evaluator.ModelEvaluatorFactory;
import org.jpmml.evaluator.NeuralNetworkEvaluator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by zhanhu on 7/15/16.
 */
public class PMMLVerifySuit {

    private static Logger logger = LoggerFactory.getLogger(PMMLVerifySuit.class);

    private String modelName;
    private String modelConfigPath;
    private String columnConfigpath;
    private String modelsPath;
    private ModelTrainConf.ALGORITHM algorithm = ModelTrainConf.ALGORITHM.NN;
    private int modelCnt;
    private String evalSetName;
    private String evalDataPath;
    private String delimiter;
    private double scoreDiff;
    private boolean isConcisePmml;

    public PMMLVerifySuit(String modelName,
                          String modelConfigPath,
                          String columnConfigpath,
                          String modelsPath,
                          int modelCnt,
                          String evalSetName,
                          String evalDataPath,
                          String delimiter,
                          double scoreDiff,
                          boolean isConcisePmml) {
        this.modelName = modelName;
        this.modelConfigPath = modelConfigPath;
        this.columnConfigpath = columnConfigpath;
        this.modelsPath = modelsPath;
        this.modelCnt = modelCnt;
        this.evalSetName = evalSetName;
        this.evalDataPath = evalDataPath;
        this.delimiter = delimiter;
        this.scoreDiff = scoreDiff;
        this.isConcisePmml = isConcisePmml;
    }

    public PMMLVerifySuit(String modelName,
                          String modelConfigPath,
                          String columnConfigpath,
                          String modelsPath,
                          ModelTrainConf.ALGORITHM algorithm,
                          int modelCnt,
                          String evalSetName,
                          String evalDataPath,
                          String delimiter,
                          double scoreDiff,
                          boolean isConcisePmml) {
        this(modelName, modelConfigPath, columnConfigpath, modelsPath, modelCnt,
                evalSetName, evalDataPath, delimiter, scoreDiff, isConcisePmml);
        this.algorithm = algorithm;
    }

    public boolean doVerification() throws Exception {
        // Step 1. Eval the scores using SHIFU
        File originModel = new File(this.modelConfigPath);
        File tmpModel = new File("ModelConfig.json");

        File originColumn = new File(this.columnConfigpath);
        File tmpColumn = new File("ColumnConfig.json");

        File modelsDir = new File(this.modelsPath);
        File tmpModelsDir = new File("models");

        FileUtils.copyFile(originModel, tmpModel);
        FileUtils.copyFile(originColumn, tmpColumn);
        FileUtils.copyDirectory(modelsDir, tmpModelsDir);

        // run evaluation set
        ShifuCLI.runEvalScore(this.evalSetName);
        File evalScore = new File("evals" + File.separator + this.evalSetName + File.separator + "EvalScore");

        Map<String, Object> params = new HashMap<String, Object>();
        params.put(ExportModelProcessor.IS_CONCISE, this.isConcisePmml);
        ShifuCLI.exportModel(null, params);

        // Step 2. Eval the scores using PMML and compare it with SHIFU output

        String DataPath = this.evalDataPath;
        String OutPath = "./pmml_out.dat";
        for (int index = 0; index < modelCnt; index++) {
            String num = Integer.toString(index);
            String pmmlPath = "pmmls" + File.separator + this.modelName + num + ".pmml";

            if ( ModelTrainConf.ALGORITHM.NN.equals(algorithm) ) {
                evalNNPmml(pmmlPath, DataPath, OutPath, this.delimiter, "model" + num);
            } else if ( ModelTrainConf.ALGORITHM.LR.equals(algorithm) ) {
                evalLRPmml(pmmlPath, DataPath, OutPath, this.delimiter, "model" + num);
            } else {
                logger.error("The algorithm - {} is not supported yet.", algorithm);
                return false;
            }

            boolean status = compareScore(evalScore, new File(OutPath), "model" + num, "\\|", this.scoreDiff);
            if ( ! status ) {
                return status;
            }

            FileUtils.deleteQuietly(new File(OutPath));
        }

        FileUtils.deleteQuietly(tmpModel);
        FileUtils.deleteQuietly(tmpColumn);
        FileUtils.deleteDirectory(tmpModelsDir);

        FileUtils.deleteQuietly(new File("./pmmls"));
        FileUtils.deleteQuietly(new File("evals"));

        return true;
    }

    private boolean compareScore(File test, File control, String scoreName, String sep, Double errorRange) throws Exception {
        List<String> testData = FileUtils.readLines(test);
        List<String> controlData = FileUtils.readLines(control);
        String[] testSchema = testData.get(0).trim().split(sep);
        String[] controlSchema = controlData.get(0).trim().split(sep);

        for (int row = 1; row < controlData.size(); row++) {
            Map<String, Object> ctx = new HashMap<String, Object>();
            Map<String, Object> controlCtx = new HashMap<String, Object>();

            String[] testRowValue = testData.get(row).split(sep, testSchema.length);
            for (int index = 0; index < testSchema.length; index++) {
                ctx.put(testSchema[index], testRowValue[index]);
            }
            String[] controlRowValue = controlData.get(row).split(sep, controlSchema.length);

            for (int index = 0; index < controlSchema.length; index++) {
                controlCtx.put(controlSchema[index], controlRowValue[index]);
            }
            Double controlScore = Double.valueOf((String) controlCtx.get(scoreName));
            Double testScore = Double.valueOf((String) ctx.get(scoreName));

            if ( Math.abs(controlScore - testScore) > errorRange ) {
                logger.error("The score doens't match {} vs {}.", controlScore, testScore);
                return false;
            }
        }

        return true;
    }

    @SuppressWarnings("unchecked")
    private void evalNNPmml(String pmmlPath, String DataPath, String OutPath, String sep, String scoreName) throws Exception {
        PMML pmml = PMMLUtils.loadPMML(pmmlPath);
        NeuralNetworkEvaluator evaluator = new NeuralNetworkEvaluator(pmml);

        PrintWriter writer = new PrintWriter(OutPath, "UTF-8");
        writer.println(scoreName);
        List<Map<FieldName, FieldValue>> input = CsvUtil.load(evaluator, DataPath, sep);

        for (Map<FieldName, FieldValue> maps : input) {
            switch (evaluator.getModel().getFunctionName()) {
                case REGRESSION:
                    Map<FieldName, Double> regressionTerm = (Map<FieldName, Double>) evaluator.evaluate(maps);
                    writer.println(regressionTerm.get(new FieldName(AbstractSpecifCreator.FINAL_RESULT)).intValue());
                    break;
                case CLASSIFICATION:
                    Map<FieldName, ClassificationMap<String>> classificationTerm = (Map<FieldName, ClassificationMap<String>>) evaluator.evaluate(maps);
                    for (ClassificationMap<String> cMap : classificationTerm.values())
                        for (Map.Entry<String, Double> entry : cMap.entrySet())
                            System.out.println(entry.getValue() * 1000);
                default:
                    break;
            }
        }

        IOUtils.closeQuietly(writer);
    }

    @SuppressWarnings("unchecked")
    private void evalLRPmml(String pmmlPath, String DataPath, String OutPath, String sep, String scoreName)
            throws Exception {
        PMML pmml = PMMLUtils.loadPMML(pmmlPath);
        Model m =pmml.getModels().get(0);
        ModelEvaluator<?> evaluator = ModelEvaluatorFactory.getInstance().getModelManager(pmml, m);
        PrintWriter writer = new PrintWriter(OutPath, "UTF-8");
        writer.println(scoreName);
        List<Map<FieldName, FieldValue>> input = CsvUtil.load(evaluator, DataPath, sep);

        for(Map<FieldName, FieldValue> maps: input) {
            Map<FieldName, Double> regressionTerm = (Map<FieldName, Double>) evaluator.evaluate(maps);
            writer.println(regressionTerm.get(new FieldName(NNSpecifCreator.FINAL_RESULT)).intValue());
        }
        IOUtils.closeQuietly(writer);
    }
}
