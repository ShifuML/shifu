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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class PMMLTranslator {

    private static final Logger LOG = LoggerFactory.getLogger(PMMLTranslator.class);

    private static final String NAME_SPACE_URI = "http://www.dmg.org/PMML-4_2";
    private static final String ELEMENT_OUT = "out";
    private static final String ELEMENT_ORIGIN = "origin";
    private static final String ZSCORE_POSTFIX = "_zscl";
    private static final String RAW_RESULT = "RawResult";
    private static final String ROUND_FUNC = "round";
    public static final String FINAL_RESULT = "FinalResult";

    private static final double EPS = 1e-10;

    private ModelConfig modelConfig;
    private List<ColumnConfig> columnConfigList;
    private List<BasicML> models;

    public PMMLTranslator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, List<BasicML> models) {
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
        this.models = models;
    }

    /**
     * Convert all models into multi pmml format
     * @return - pmmls for models
     */
    public List<PMML> translate() {
        List<PMML> pmmls = new ArrayList<PMML>(models.size());

        for(int index = 0; index < models.size(); index++) {
            pmmls.add(translate(index));
        }

        return pmmls;
    }

    /**
     * Convert some model into pmml format
     * @param index - which model to pmml format
     * @return  pmml for model
     *          Notice, if the index is out of bound, return null
     */
    public PMML translate(int index) {
        if ( index > models.size() ) {
            // out-of-bound. return null or throw exception, which is better
            return null;
        }

        PMML pmml = new PMML();

        // create and set data dictionary
        pmml.setDataDictionary(createDataDictionary(columnConfigList));

        // create model element
        Model model = createModel(modelConfig);

        // create mining schema
        model.setMiningSchema(createModelMiningSchema(columnConfigList));

        // create variable statistical info
        model.setModelStats(createModelMiningStats(columnConfigList));

        // create variable transform
        model.setLocalTransformations(createLocalTransformations(columnConfigList, modelConfig.getNormalizeStdDevCutOff()));

        // create specification
        if ( model instanceof  NeuralNetwork ) {
            NeuralNetwork nnPmmlModel = (NeuralNetwork) model;
            new PMMLEncogNeuralNetworkModel().adaptMLModelToPMML((BasicNetwork) models.get(index), nnPmmlModel);

            nnPmmlModel.withOutput(createNormalizedOutput());
        } else {
            // something wrong
            throw new RuntimeException("Not support model type.");
        }

        pmml.withModels(model);

        return pmml;
    }

    /**
     * Convert the list of @ColumnConfig into data dictionary
     * @param columnConfigList - ColumnConfig list from Shifu
     * @return @DataDictionary that represent ColumnConfig list
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
     * Create a @Model according @ModelConfig. Currently we only support NeuralNetwork
     * @param modelConfig @ModelConfig from Shifu
     * @return a model with targets
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
     * Create model mining schema from ColumnConfig list.
     * Only final select and target column will be added into MiningSchema
     * @param columnConfigList List of @ColumnConfig from Shifu
     * @return MiningSchema for model
     */
    private MiningSchema createModelMiningSchema(List<ColumnConfig> columnConfigList) {
        MiningSchema miningSchema = new MiningSchema();

        for (ColumnConfig columnConfig : columnConfigList) {
            if ( columnConfig.isFinalSelect() || columnConfig.isTarget() ) {
                MiningField miningField = new MiningField();

                miningField.setName(FieldName.create(columnConfig.getColumnName()));
                miningField.setOptype(getOptype(columnConfig));

                if ( columnConfig.isFinalSelect() ) {
                    miningField.setUsageType(FieldUsageType.ACTIVE);
                } else if (columnConfig.isTarget()) {
                    miningField.setUsageType(FieldUsageType.TARGET);
                }

                miningSchema.withMiningFields(miningField);
            }

        }

        return miningSchema;
    }

    /**
     * Create ModelStats for model. The needed info are all from ColumnConfig list
     * Only final select column will be added into ModelStats
     * @param columnConfigList List of @ColumnConfig from Shifu
     * @return ModelStats for model
     */
    private ModelStats createModelMiningStats(List<ColumnConfig> columnConfigList) {
        ModelStats modelStats = new ModelStats();

        for (ColumnConfig columnConfig : columnConfigList) {
            if ( columnConfig.isFinalSelect() ) {
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
     * Create LocalTransformations for model. The needed info are all from ColumnConfig list
     * Only final select column will be added into LocalTransformations
     * @param columnConfigList List of @ColumnConfig from Shifu
     * @return LocalTransformations for model
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
     * Create @DerivedField for categorical variable
     * @param config - ColumnConfig for categorical variable
     * @param cutoff - cutoff for normalization
     * @return DerivedField for variable
     */
    private DerivedField createCategoricalDerivedField(ColumnConfig config, double cutoff) {
        Document document = null;
        try {
            document = DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();
        } catch (ParserConfigurationException e) {
            LOG.error("Fail to create document node.", e);
            throw new RuntimeException("Fail to create document node.", e);
        }

        String defaultValue = "0.0";
        String missingValue = "0.0";

        InlineTable inlineTable = new InlineTable();
        for (int i = 0; i < config.getBinCategory().size(); i++) {
            String cval = config.getBinCategory().get(i);
            String dval = Normalizer.normalize(config, cval, cutoff).toString();

            Element out = document.createElementNS(NAME_SPACE_URI, ELEMENT_OUT);
            out.setTextContent(dval);

            Element origin = document.createElementNS(NAME_SPACE_URI, ELEMENT_ORIGIN);
            origin.setTextContent(cval);

            inlineTable.withRows(new Row().withContent(origin).withContent(out));
            if ( StringUtils.isBlank(cval) ){
                missingValue = dval;
            }
        }

        MapValues mapValues = new MapValues("out").withDataType(DataType.DOUBLE)
                .withDefaultValue(defaultValue)
                .withFieldColumnPairs(new FieldColumnPair(new FieldName(config.getColumnName()), ELEMENT_ORIGIN))
                .withInlineTable(inlineTable)
                .withMapMissingTo(missingValue);

        return new DerivedField(OpType.CONTINUOUS, DataType.DOUBLE)
                .withName(FieldName.create(config.getColumnName() + ZSCORE_POSTFIX))
                .withExpression(mapValues);
    }

    /**
     * Create @DerivedField for numerical variable
     * @param config - ColumnConfig for numerical variable
     * @param cutoff - cutoff of normalization
     * @return DerivedField for variable
     */
    private DerivedField createNumericalDerivedField(ColumnConfig config, double cutoff) {

        //added capping logic to linearNorm
        LinearNorm from = new LinearNorm().withOrig(config.getMean() - config.getStdDev() * cutoff).withNorm(-cutoff);
        LinearNorm to = new LinearNorm().withOrig(config.getMean() + config.getStdDev() * cutoff).withNorm(cutoff);
        NormContinuous normContinuous = new NormContinuous(FieldName.create(config.getColumnName())).
                withLinearNorms(from, to).withMapMissingTo(0.0).withOutliers(OutlierTreatmentMethodType.AS_EXTREME_VALUES);

        //derived field name is consisted of FieldName and "_zscl"
        return new DerivedField(OpType.CONTINUOUS, DataType.DOUBLE).
                withName(FieldName.create(config.getColumnName() + ZSCORE_POSTFIX)).withExpression(normContinuous);
    }

    /**
     * Create @ConStats for numerical variable
     * @param columnConfig - ColumnConfig to generate ConStats
     * @return ConStats for variable
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
     * Generate Woe data from positive and negative counts
     * @param binCountPos - positive count list
     * @param binCountNeg - negative count list
     * @return Woe value list
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
     * Create @NumericInfo for numerical variable
     * @param columnConfig - ColumnConfig for numerical variable
     * @return NumericInfo for variable
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
     * Create common extension list from ColumnConfig
     * @param columnConfig - ColumnConfig to create extension
     * @return extension list
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
     * Create extension list from HashMap
     * @param extensionMap the <String,String> map to create extension list
     * @return extension list
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
     * Create @Array for numerical variable
     * @param columnConfig - ColumnConfig for numerical variable
     * @return Array for numerical variable ( positive count + negative count )
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
     * Get @OpType from ColumnConfig
     *      Meta Column -> ORDINAL
     *      Target Column -> CATEGORICAL
     *      Categorical Column -> CATEGORICAL
     *      Numerical Column -> CONTINUOUS
     * @param columnConfig - ColumnConfig for variable
     * @return OpType
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
     * Get DataType from OpType
     *      CONTINUOUS -> DOUBLE
     *      Other -> STRING
     * @param optype OpType
     * @return DataType
     */
    private DataType getDataType(OpType optype) {
        return (optype.equals(OpType.CONTINUOUS) ? DataType.DOUBLE : DataType.STRING);
    }

    /**
     * Create @Targets from @ModelConfig
     * @param modelConfig - ModelConfig from Shifu
     * @return Targets that includes positive tags and negative tags
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

            targetValueList.add(neg);
        }

        target.withTargetValues(targetValueList);

        targets.withTargets(target);

        return targets;
    }

    /**
     * Create the normalized output for model, since the final score should be 0 ~ 1000, instead of 0.o ~ 1.0
     * @return @Output for model
     */
    private Output createNormalizedOutput() {
        Output output = new Output();

        output.withOutputFields(
                createOutputField(RAW_RESULT, OpType.CONTINUOUS, DataType.DOUBLE, ResultFeatureType.PREDICTED_VALUE));

        OutputField finalResult = createOutputField(FINAL_RESULT,
                OpType.CONTINUOUS, DataType.DOUBLE, ResultFeatureType.TRANSFORMED_VALUE);
        finalResult.withExpression(createApplyFunc());

        output.withOutputFields(finalResult);

        return output;
    }

    /**
     * Create the output field, and set the field name, operation type, data type and feature type
     * @param fieldName - the name of output field
     * @param opType - operation type
     * @param dataType - data type
     * @param feature - result feature type
     * @return @OutputField
     */
    private OutputField createOutputField(String fieldName, OpType opType, DataType dataType, ResultFeatureType feature) {
        OutputField outputField = new OutputField();
        outputField.withName(new FieldName(fieldName));
        outputField.withOptype(opType);
        outputField.withDataType(dataType);
        outputField.withFeature(feature);
        return outputField;
    }

    /**
     * Create the apply expression for final output, the function is "round"
     * @return @Apply
     */
    private Apply createApplyFunc() {
        Apply apply = new Apply();

        apply.withFunction(ROUND_FUNC);

        NormContinuous normContinuous = new NormContinuous();
        normContinuous.withField(new FieldName(RAW_RESULT));
        normContinuous.withLinearNorms(new LinearNorm().withOrig(0).withNorm(0));
        normContinuous.withLinearNorms(new LinearNorm().withOrig(1).withNorm(1000));

        apply.withExpressions(normContinuous);

        return apply;
    }
}
