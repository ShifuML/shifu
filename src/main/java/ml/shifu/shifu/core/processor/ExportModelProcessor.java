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
package ml.shifu.shifu.core.processor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.pmml.PMMLEncogNeuralNetworkModel;
import ml.shifu.shifu.core.pmml.PMMLUtils;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.core.Normalizer;

import org.apache.commons.lang.StringUtils;
import org.dmg.pmml.Array;
import org.dmg.pmml.ContStats;
import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.DiscrStats;
import org.dmg.pmml.Extension;
import org.dmg.pmml.FieldColumnPair;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.InlineTable;
import org.dmg.pmml.Interval;
import org.dmg.pmml.LinearNorm;
import org.dmg.pmml.LocalTransformations;
import org.dmg.pmml.MapValues;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Model;
import org.dmg.pmml.ModelStats;
import org.dmg.pmml.NeuralNetwork;
import org.dmg.pmml.NormContinuous;
import org.dmg.pmml.NumericInfo;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMML;
import org.dmg.pmml.Row;
import org.dmg.pmml.Target;
import org.dmg.pmml.TargetValue;
import org.dmg.pmml.Targets;
import org.dmg.pmml.UnivariateStats;
import org.encog.ml.BasicML;
import org.encog.neural.networks.BasicNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ExportModelProcessor class
 * 
 * @author zhanhu
 * @Nov 6, 2014
 *
 */
public class ExportModelProcessor extends BasicModelProcessor implements Processor {
    
    public static final String PMML = "pmml";
    public static final double EPS = 1e-10;
    
    /**
     * log object
     */
    private final static Logger log = LoggerFactory.getLogger(ExportModelProcessor.class);
    
    
    private String type;
    
    public ExportModelProcessor(String type) {
        this.type = type;
    }

    /* (non-Javadoc)
     * @see ml.shifu.shifu.core.processor.Processor#run()
     */
    @Override
    public int run() throws Exception {
        if ( StringUtils.isBlank(type) ) {
            type = PMML;
        }
        
        if ( !type.equalsIgnoreCase(PMML) ) {
            log.error("Unsupported output format - " + type);
            return -1;
        }
        
        ModelConfig modelConfig = CommonUtils.loadModelConfig();
        List<ColumnConfig> columnConfigList = CommonUtils.loadColumnConfigList();
        
        PathFinder pathFinder = new PathFinder(modelConfig);
        List<BasicML> models = CommonUtils.loadBasicModels(pathFinder.getModelsPath(SourceType.LOCAL), ALGORITHM.NN);
        
        PMML pmml = new PMML();
        
        // create and set data dictionary 
        pmml.setDataDictionary(createDataDictionary(columnConfigList));
        
        // create model element
        pmml.withModels(createModel(modelConfig));
        
        // create mining schema
        for ( Model model : pmml.getModels() ) {
            model.setMiningSchema(createModelMiningSchema(columnConfigList));
        }
        
        // create variable statistical info
        for ( Model model : pmml.getModels() ) {
            model.setModelStats(createModelMiningStats(columnConfigList));
        }
        
        // create variable transform 
        for ( Model model : pmml.getModels() ) {
            model.setLocalTransformations(createLocalTransformations(columnConfigList));
        }
        
        // create specification
        for ( Model model : pmml.getModels() ) {
            NeuralNetwork nnPmmlModel = (NeuralNetwork) model;
            nnPmmlModel = new PMMLEncogNeuralNetworkModel().adaptMLModelToPMML((BasicNetwork)models.get(0), nnPmmlModel);
            
            pmml.getModels().set(0, nnPmmlModel);
        }
        
        PMMLUtils.savePMML(pmml, modelConfig.getModelSetName() + ".pmml");
        
        return 0;
    }


    /**
     * @param columnConfigList
     * @return
     */
    private DataDictionary createDataDictionary(List<ColumnConfig> columnConfigList) {
        DataDictionary dict = new DataDictionary();
        
        List<DataField> fields = new ArrayList<DataField>();
        
        for ( ColumnConfig columnConfig : columnConfigList ) {
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
        if ( ALGORITHM.NN.name().equalsIgnoreCase(modelConfig.getTrain().getAlgorithm()) ) {
            model = new NeuralNetwork();
        } else {
            throw new RuntimeException("Model not supported: " + modelConfig.getTrain().getAlgorithm());
        }
        
        model.setTargets(createTargets(modelConfig));
        
        return model;
    }
    
    /**
     * @param columnConfigList
     * @return
     */
    private MiningSchema createModelMiningSchema(List<ColumnConfig> columnConfigList) {
        MiningSchema miningSchema = new MiningSchema();
        
        for ( ColumnConfig columnConfig : columnConfigList ) {
            MiningField miningField = new MiningField();
            
            miningField.setName(FieldName.create(columnConfig.getColumnName()));
            miningField.setOptype(getOptype(columnConfig));
            
            if ( columnConfig.isForceSelect() ) {
                miningField.setUsageType(FieldUsageType.ACTIVE);
            } else if ( columnConfig.isTarget() ) {
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
        
        for ( ColumnConfig columnConfig : columnConfigList ) {
            if ( columnConfig.isFinalSelect() ) {
                UnivariateStats univariateStats = new UnivariateStats();
                univariateStats.setField(FieldName.create(columnConfig.getColumnName()));
            
                if ( columnConfig.isCategorical() ) {
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
    private LocalTransformations createLocalTransformations(List<ColumnConfig> columnConfigList) {
        LocalTransformations localTransformations = new LocalTransformations();
        
        for ( ColumnConfig columnConfig : columnConfigList ) {
            if ( columnConfig.isFinalSelect() ) {
                
                DerivedField derivedField = null;
                
                if ( columnConfig.isCategorical() ) { 
                    derivedField = createCategoricalDerivedField(columnConfig);
                } else {
                    derivedField = createNumericalDerivedField(columnConfig);
                }
                
                localTransformations.withDerivedFields(derivedField);
            }
        }
        
        return localTransformations;
    }
    
    /**
     * @param columnConfig
     * @return
     */
    private DerivedField createCategoricalDerivedField(ColumnConfig columnConfig) {
        DerivedField derivedField = new DerivedField();
        
        derivedField.setName(FieldName.create(columnConfig.getColumnName()));
        derivedField.setOptype(OpType.CONTINUOUS);
        derivedField.setDataType(DataType.DOUBLE);
        
        MapValues mapValues = new MapValues();
        mapValues.setDataType(DataType.DOUBLE);
        //TODO. What to use?
        mapValues.setMapMissingTo("");
        mapValues.setOutputColumn("out");
        
        FieldColumnPair fieldColumnPair = new FieldColumnPair();
        fieldColumnPair.setField(FieldName.create(columnConfig.getColumnName()));
        fieldColumnPair.setColumn("origin");
        
        mapValues.withFieldColumnPairs(fieldColumnPair);
        
        InlineTable inlineTable = new InlineTable();
        for ( int i = 0; i < columnConfig.getBinCategory().size(); i ++ ) {
            String cval = columnConfig.getBinCategory().get(i);
            double dval = Normalizer.normalize(columnConfig, cval);
            
            Row row = new Row();
            row.withContent(cval);
            row.withContent(dval);
            
            inlineTable.withRows(row);
        }
        
        mapValues.withInlineTable(inlineTable);
        
        derivedField.setExpression(mapValues);
        
        return derivedField;
        
    }

    /**
     * @param columnConfig
     * @return
     */
    private DerivedField createNumericalDerivedField(ColumnConfig columnConfig) {
        DerivedField derivedField = new DerivedField();
        
        derivedField.setName(FieldName.create(columnConfig.getColumnName()));
        derivedField.setOptype(OpType.CONTINUOUS);
        derivedField.setDataType(DataType.DOUBLE);
        
        NormContinuous normContinuous = new NormContinuous();
        normContinuous.setField(FieldName.create(columnConfig.getColumnName()));
        
        LinearNorm fromNorm = new LinearNorm();
        fromNorm.setOrig(0);
        fromNorm.setNorm(-1 * columnConfig.getMean() / columnConfig.getStdDev());
        
        LinearNorm toNorm = new LinearNorm();
        toNorm.setOrig(columnConfig.getMean());
        toNorm.setNorm(0);
        
        normContinuous.withLinearNorms(fromNorm, toNorm);
        
        derivedField.setExpression(normContinuous);
        
        return derivedField;
    }

    /**
     * @param columnConfig
     * @return
     */
    private ContStats createConStats(ColumnConfig columnConfig) {
        ContStats conStats = new ContStats();
        
        List<Interval> intervals = new ArrayList<Interval>();
        for ( int i = 0; i < columnConfig.getBinBoundary().size(); i ++ ) {
            Interval interval = new Interval();
            interval.setClosure(Interval.Closure.OPEN_CLOSED);
            interval.setLeftMargin(columnConfig.getBinBoundary().get(i));
            
            if ( i == columnConfig.getBinBoundary().size() - 1 ) {
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
        
        for ( int i = 0; i < binCountPos.size(); i ++ ) {
            sumPos += binCountPos.get(i);
            sumNeg += binCountNeg.get(i);
        }
        
        for ( int i = 0; i < binCountPos.size(); i ++ ) {
            woe.add(Math.log( (binCountPos.get(i) / sumPos + EPS) / (binCountNeg.get(i) / sumNeg + EPS) ) );
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

        // numericInfo.setInterQuartileRange(value);
        // sort value for small to large, interQuartileRange = value(75%) - value(25%) 
        
        // numericInfo.withQuantiles(values);
        //        for (int i = 0; i < num; i++) {
        //            Quantile quantile = new Quantile();
        //            quantile.setQuantileLimit(((double) i / (num - 1)) * 100);
        //            quantile.setQuantileValue(values.get((size - 1) * i / (num - 1)));
        //            quantiles.add(quantile);
        //        }
        
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
        
        for ( String key : extensionMap.keySet() ) {
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
        for ( int i = 0; i < binCountAll.size(); i ++ ) {
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
        if ( columnConfig.isMeta() ) {
            return OpType.ORDINAL;
        } else if ( columnConfig.isTarget() ) {
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
     * @param modelConfig
     * @return
     */
    public Targets createTargets(ModelConfig modelConfig) {
        Targets targets = new Targets();

        Target target = new Target();

        target.setOptype(OpType.CATEGORICAL);
        target.setField(new FieldName(modelConfig.getTargetColumnName()));

        List<TargetValue> targetValueList = new ArrayList<TargetValue>();
        
        for ( String posTagValue : modelConfig.getPosTags() ) {
            TargetValue pos = new TargetValue();
            pos.setValue(posTagValue);
            pos.setDisplayValue("Positive");
            
            targetValueList.add(pos);
        }
        
        for ( String negTagValue : modelConfig.getNegTags() ) {
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
