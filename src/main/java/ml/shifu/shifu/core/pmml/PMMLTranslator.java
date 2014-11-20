/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.pmml;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf;
import ml.shifu.shifu.core.Normalizer;
import org.apache.commons.lang.StringUtils;
import org.dmg.pmml.*;
import org.encog.ml.BasicML;
import org.encog.neural.networks.BasicNetwork;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class PMMLTranslator {

    private ModelConfig modelConfig;
    private List<ColumnConfig> columnConfigList;
    private List<BasicML> models;

    private static final String NAMESPACEURI = "http://www.dmg.org/PMML-4_2";
    private static final String ELEMENTOUT = "out";
    private static final String ELEMENTORIGIN = "origin";
    public static final double EPS = 1e-10;


    public PMMLTranslator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, List<BasicML> models) {
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
        this.models = models;
    }

    public PMML translate() {
        return this.translate(0);
    }

    public PMML translate(int index) {
        PMML pmml = new PMML();

        // create and set data dictionary
        pmml.setDataDictionary(createDataDictionary(columnConfigList));

        // create model element
        pmml.withModels(createModel(modelConfig));

        // create mining schema
        for (Model model : pmml.getModels()) {
            model.setMiningSchema(createModelMiningSchema(columnConfigList));
        }

        // create variable statistical info
        for (Model model : pmml.getModels()) {
            model.setModelStats(createModelMiningStats(columnConfigList));
        }

        // create variable transform
        for (Model model : pmml.getModels()) {
            model.setLocalTransformations(createLocalTransformations(columnConfigList, modelConfig.getNormalizeStdDevCutOff()));
        }

        // create specification
        for (Model model : pmml.getModels()) {
            NeuralNetwork nnPmmlModel = (NeuralNetwork) model;
            nnPmmlModel = new PMMLEncogNeuralNetworkModel().adaptMLModelToPMML((BasicNetwork) models.get(index), nnPmmlModel);
            pmml.getModels().set(0, nnPmmlModel);
        }
        return pmml;
    }


    /**
     * @param columnConfigList
     * @return
     */
    private DataDictionary createDataDictionary(List<ColumnConfig> columnConfigList) {
        DataDictionary dict = new DataDictionary();

        List<DataField> fields = new ArrayList<DataField>();

        for (ColumnConfig columnConfig : columnConfigList) {
            DataField field = new DataField();
            field.setName(FieldName.create(columnConfig.getColumnName()));
            field.setOptype(getOptype(columnConfig));
            field.setDataType(getDataType(field.getOptype()));

            fields.add(field);
        }

        dict.withDataFields(fields);
        dict.withNumberOfFields(fields.size());

        return dict;
    }


    /**
     * @param modelConfig
     * @return
     */
    private Model createModel(ModelConfig modelConfig) {
        Model model = null;
        if (ModelTrainConf.ALGORITHM.NN.name().equalsIgnoreCase(modelConfig.getTrain().getAlgorithm())) {
            model = new NeuralNetwork();
        } else {
            throw new RuntimeException("Model not supported: " + modelConfig.getTrain().getAlgorithm());
        }
        model.setTargets(createTargets(modelConfig));
        return model;
    }

    /**
     * @param modelConfig
     * @return
     */
    private List<Model> createModels(ModelConfig modelConfig, int size) {
        List<Model> models = new ArrayList<Model>(size);
        if (ModelTrainConf.ALGORITHM.NN.name().equalsIgnoreCase(modelConfig.getTrain().getAlgorithm())) {
            for (int i = 0; i < size; i++) {
                models.add(new NeuralNetwork());
                models.get(i).setTargets(createTargets(modelConfig));
            }
        } else {
            throw new RuntimeException("Model not supported: " + modelConfig.getTrain().getAlgorithm());
        }
        return models;
    }

    /**
     * @param columnConfigList
     * @return
     */
    private MiningSchema createModelMiningSchema(List<ColumnConfig> columnConfigList) {
        MiningSchema miningSchema = new MiningSchema();

        for (ColumnConfig columnConfig : columnConfigList) {
            MiningField miningField = new MiningField();

            miningField.setName(FieldName.create(columnConfig.getColumnName()));
            miningField.setOptype(getOptype(columnConfig));

            if (columnConfig.isForceSelect()) {
                miningField.setUsageType(FieldUsageType.ACTIVE);
            } else if (columnConfig.isTarget()) {
                miningField.setUsageType(FieldUsageType.TARGET);
            }

            miningSchema.withMiningFields(miningField);
        }

        return miningSchema;
    }

    /**
     * @param columnConfigList
     * @return
     */
    private ModelStats createModelMiningStats(List<ColumnConfig> columnConfigList) {
        ModelStats modelStats = new ModelStats();

        for (ColumnConfig columnConfig : columnConfigList) {
            if (columnConfig.isFinalSelect()) {
                UnivariateStats univariateStats = new UnivariateStats();
                univariateStats.setField(FieldName.create(columnConfig.getColumnName()));

                if (columnConfig.isCategorical()) {
                    DiscrStats discrStats = new DiscrStats();

                    Array countArray = createCountArray(columnConfig);
                    discrStats.withArrays(countArray);

                    List<Extension> extensions = createExtensions(columnConfig);
                    discrStats.withExtensions(extensions);

                    univariateStats.setDiscrStats(discrStats);
                } else { // numerical column
                    univariateStats.setNumericInfo(createNumericInfo(columnConfig));
                    univariateStats.setContStats(createConStats(columnConfig));
                }

                modelStats.withUnivariateStats(univariateStats);
            }
        }

        return modelStats;
    }

    /**
     * @param columnConfigList
     * @return
     */
    private LocalTransformations createLocalTransformations(List<ColumnConfig> columnConfigList, double cutoff) {

        LocalTransformations localTransformations = new LocalTransformations();
        for (ColumnConfig config : columnConfigList) {
            if (config.isFinalSelect()) {
                localTransformations.withDerivedFields(
                        config.isCategorical() ? createCategoricalDerivedField(config, cutoff) : createNumericalDerivedField(config, cutoff)
                );
            }
        }
        return localTransformations;
    }

    /**
     * @param config
     * @return
     */
    private DerivedField createCategoricalDerivedField(ColumnConfig config, double cutoff) {

        InlineTable inlineTable = new InlineTable();
        for (int i = 0; i < config.getBinCategory().size(); i++) {
            String cval = config.getBinCategory().get(i);
            String dval = Normalizer.normalize(config, cval, cutoff).toString();
            Document document = null;
            try {
                document = DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();

            } catch (ParserConfigurationException e) {
                e.printStackTrace();
            }
            Element out = document.createElementNS(NAMESPACEURI, ELEMENTOUT);
            out.setTextContent(dval);
            Element origin = document.createElementNS(NAMESPACEURI, ELEMENTORIGIN);
            origin.setTextContent(cval);
            inlineTable.withRows(new Row().withContent(origin).withContent(out));
        }

        MapValues mapValues = new MapValues("out").withDataType(DataType.DOUBLE).withDefaultValue("0.0").
                withFieldColumnPairs(new FieldColumnPair(new FieldName(config.getColumnName()), "origin")).
                withInlineTable(inlineTable);
        mapValues.setMapMissingTo("0.0");

        return new DerivedField(OpType.CONTINUOUS, DataType.DOUBLE).
                withName(FieldName.create(config.getColumnName() + "_zscl")).withExpression(mapValues);
    }

    /**
     * @param config
     * @return derivedField
     */
    private DerivedField createNumericalDerivedField(ColumnConfig config, double cutoff) {

        //added capping logic to linearNorm
        LinearNorm from = new LinearNorm().withOrig(config.getMean() - config.getStdDev() * cutoff).withNorm(-cutoff);
        LinearNorm to = new LinearNorm().withOrig(config.getMean() + config.getStdDev() * cutoff).withNorm(cutoff);
        NormContinuous normContinuous = new NormContinuous(FieldName.create(config.getColumnName())).
                withLinearNorms(from, to).withMapMissingTo(0.0).withOutliers(OutlierTreatmentMethodType.AS_EXTREME_VALUES);

        //derived field name is consisted of FieldName and "_zscl"
        return new DerivedField(OpType.CONTINUOUS, DataType.DOUBLE).
                withName(FieldName.create(config.getColumnName() + "_zscl")).withExpression(normContinuous);
    }

    /**
     * @param columnConfig
     * @return
     */
    private ContStats createConStats(ColumnConfig columnConfig) {
        ContStats conStats = new ContStats();

        List<Interval> intervals = new ArrayList<Interval>();
        for (int i = 0; i < columnConfig.getBinBoundary().size(); i++) {
            Interval interval = new Interval();
            interval.setClosure(Interval.Closure.OPEN_CLOSED);
            interval.setLeftMargin(columnConfig.getBinBoundary().get(i));

            if (i == columnConfig.getBinBoundary().size() - 1) {
                interval.setRightMargin(Double.POSITIVE_INFINITY);
            } else {
                interval.setRightMargin(columnConfig.getBinBoundary().get(i + 1));
            }

            intervals.add(interval);
        }
        conStats.withIntervals(intervals);

        Map<String, String> extensionMap = new HashMap<String, String>();

        extensionMap.put("BinCountPos", columnConfig.getBinCountPos().toString());
        extensionMap.put("BinCountNeg", columnConfig.getBinCountNeg().toString());
        extensionMap.put("BinWeightedCountPos", columnConfig.getBinWeightedPos().toString());
        extensionMap.put("BinWeightedCountNeg", columnConfig.getBinWeightedNeg().toString());
        extensionMap.put("BinPosRate", columnConfig.getBinPosRate().toString());
        extensionMap.put("BinWOE", calculateWoe(columnConfig.getBinCountPos(), columnConfig.getBinCountNeg()).toString());
        extensionMap.put("KS", Double.toString(columnConfig.getKs()));
        extensionMap.put("IV", Double.toString(columnConfig.getIv()));
        conStats.withExtensions(createExtensions(extensionMap));

        return conStats;
    }

    /**
     * @param binCountPos
     * @param binCountNeg
     * @return
     */
    private List<Double> calculateWoe(List<Integer> binCountPos, List<Integer> binCountNeg) {
        List<Double> woe = new ArrayList<Double>();

        double sumPos = 0.0;
        double sumNeg = 0.0;

        for (int i = 0; i < binCountPos.size(); i++) {
            sumPos += binCountPos.get(i);
            sumNeg += binCountNeg.get(i);
        }

        for (int i = 0; i < binCountPos.size(); i++) {
            woe.add(Math.log((binCountPos.get(i) / sumPos + EPS) / (binCountNeg.get(i) / sumNeg + EPS)));
        }

        return woe;
    }

    /**
     * @param columnConfig
     * @return
     */
    private NumericInfo createNumericInfo(ColumnConfig columnConfig) {
        NumericInfo numericInfo = new NumericInfo();

        numericInfo.setMaximum(columnConfig.getColumnStats().getMax());
        numericInfo.setMinimum(columnConfig.getColumnStats().getMin());
        numericInfo.setMean(columnConfig.getMean());
        numericInfo.setMedian(columnConfig.getMedian());
        numericInfo.setStandardDeviation(columnConfig.getStdDev());

        return numericInfo;
    }

    /**
     * @param columnConfig
     * @return
     */
    private List<Extension> createExtensions(ColumnConfig columnConfig) {
        Map<String, String> extensionMap = new HashMap<String, String>();

        extensionMap.put("BinCountPos", columnConfig.getBinCountPos().toString());
        extensionMap.put("BinCountNeg", columnConfig.getBinCountNeg().toString());
        extensionMap.put("BinWeightedCountPos", columnConfig.getBinWeightedPos().toString());
        extensionMap.put("BinWeightedCountNeg", columnConfig.getBinWeightedNeg().toString());
        extensionMap.put("BinPosRate", columnConfig.getBinPosRate().toString());

        return createExtensions(extensionMap);
    }

    /**
     * @param extensionMap
     * @return
     */
    private List<Extension> createExtensions(Map<String, String> extensionMap) {
        List<Extension> extensions = new ArrayList<Extension>();

        for (String key : extensionMap.keySet()) {
            Extension extension = new Extension();
            extension.setName(key);
            extension.setValue(extensionMap.get(key));

            extensions.add(extension);
        }

        return extensions;
    }

    /**
     * @param columnConfig
     * @return
     */
    private Array createCountArray(ColumnConfig columnConfig) {
        Array countAllArray = new Array();

        List<Integer> binCountAll = new ArrayList<Integer>(columnConfig.getBinCountPos().size());
        for (int i = 0; i < binCountAll.size(); i++) {
            binCountAll.add(columnConfig.getBinCountPos().get(i) + columnConfig.getBinCountNeg().get(i));
        }

        countAllArray.setType(Array.Type.INT);
        countAllArray.setN(binCountAll.size());
        countAllArray.setValue(StringUtils.join(binCountAll, ' '));

        return countAllArray;
    }

    /**
     * @param columnConfig
     * @return
     */
    private OpType getOptype(ColumnConfig columnConfig) {
        if (columnConfig.isMeta()) {
            return OpType.ORDINAL;
        } else if (columnConfig.isTarget()) {
            return OpType.CATEGORICAL;
        } else {
            return (columnConfig.isCategorical() ? OpType.CATEGORICAL : OpType.CONTINUOUS);
        }
    }

    /**
     * @param optype
     * @return
     */
    private DataType getDataType(OpType optype) {
        return (optype.equals(OpType.CONTINUOUS) ? DataType.DOUBLE : DataType.STRING);
    }

    /**
     * Create target from @ModelConfig
     *
     * @param modelConfig
     * @return
     */
    public Targets createTargets(ModelConfig modelConfig) {
        Targets targets = new Targets();

        Target target = new Target();

        target.setOptype(OpType.CATEGORICAL);
        target.setField(new FieldName(modelConfig.getTargetColumnName()));

        List<TargetValue> targetValueList = new ArrayList<TargetValue>();

        for (String posTagValue : modelConfig.getPosTags()) {
            TargetValue pos = new TargetValue();
            pos.setValue(posTagValue);
            pos.setDisplayValue("Positive");

            targetValueList.add(pos);
        }

        for (String negTagValue : modelConfig.getNegTags()) {
            TargetValue neg = new TargetValue();
            neg.setValue(negTagValue);
            neg.setDisplayValue("Negative");

            target.withTargetValues(neg);
        }

        target.withTargetValues(targetValueList);

        targets.withTargets(target);

        return targets;
    }
}
