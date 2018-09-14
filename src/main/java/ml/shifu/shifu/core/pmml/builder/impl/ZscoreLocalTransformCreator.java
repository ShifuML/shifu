/*
 * Copyright [2013-2016] PayPal Software Foundation
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
package ml.shifu.shifu.core.pmml.builder.impl;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatNetwork;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractPmmlElementCreator;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.collections.CollectionUtils;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldColumnPair;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.InlineTable;
import org.dmg.pmml.LinearNorm;
import org.dmg.pmml.LocalTransformations;
import org.dmg.pmml.MapValues;
import org.dmg.pmml.NormContinuous;
import org.dmg.pmml.OpType;
import org.dmg.pmml.OutlierTreatmentMethodType;
import org.dmg.pmml.Row;
import org.encog.ml.BasicML;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

/**
 * Created by zhanhu on 3/29/16.
 */
public class ZscoreLocalTransformCreator extends AbstractPmmlElementCreator<LocalTransformations> {

    private static final Logger LOG = LoggerFactory.getLogger(ZscoreLocalTransformCreator.class);

    protected static final String NAME_SPACE_URI = "http://www.dmg.org/PMML-4_2";
    protected static final String ELEMENT_OUT = "out";
    protected static final String ELEMENT_ORIGIN = "origin";

    public ZscoreLocalTransformCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        super(modelConfig, columnConfigList);
    }

    public ZscoreLocalTransformCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, boolean isConcise) {
        super(modelConfig, columnConfigList, isConcise);
    }

    @Override
    public LocalTransformations build(BasicML basicML) {
        LocalTransformations localTransformations = new LocalTransformations();

        if(basicML instanceof BasicFloatNetwork) {
            BasicFloatNetwork bfn = (BasicFloatNetwork) basicML;
            Set<Integer> featureSet = bfn.getFeatureSet();
            for(ColumnConfig config: columnConfigList) {
                if(config.isFinalSelect()
                        && (CollectionUtils.isEmpty(featureSet) || featureSet.contains(config.getColumnNum()))) {
                    double cutoff = modelConfig.getNormalizeStdDevCutOff();
                    localTransformations.withDerivedFields(config.isCategorical() ? createCategoricalDerivedField(
                            config, cutoff, modelConfig.getNormalizeType()) : createNumericalDerivedField(config,
                            cutoff, modelConfig.getNormalizeType()));
                }
            }
        } else {
            for(ColumnConfig config: columnConfigList) {
                if(config.isFinalSelect()) {
                    double cutoff = modelConfig.getNormalizeStdDevCutOff();
                    localTransformations.withDerivedFields(config.isCategorical() ? createCategoricalDerivedField(
                            config, cutoff, modelConfig.getNormalizeType()) : createNumericalDerivedField(config,
                            cutoff, modelConfig.getNormalizeType()));
                }
            }
        }
        return localTransformations;
    }

    /**
     * Create DerivedField for categorical variable
     * 
     * @param config
     *            - ColumnConfig for categorical variable
     * @param cutoff
     *            - cutoff for normalization
     * @param normType
     *            - the normalization method that is used to generate DerivedField
     * @return DerivedField for variable
     */
    protected List<DerivedField> createCategoricalDerivedField(ColumnConfig config, double cutoff,
            ModelNormalizeConf.NormType normType) {
        Document document = null;
        try {
            document = DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();
        } catch (ParserConfigurationException e) {
            LOG.error("Fail to create document node.", e);
            throw new RuntimeException("Fail to create document node.", e);
        }

        String defaultValue = Normalizer.normalize(config, "doesn't exist at all...by paypal", cutoff, normType).get(0)
                .toString();
        String missingValue = Normalizer.normalize(config, null, cutoff, normType).get(0).toString();

        InlineTable inlineTable = new InlineTable();
        for(int i = 0; i < config.getBinCategory().size(); i++) {
            List<String> catVals = CommonUtils.flattenCatValGrp(config.getBinCategory().get(i));
            for(String cval: catVals) {
                String dval = Normalizer.normalize(config, cval, cutoff, normType).get(0).toString();

                Element out = document.createElementNS(NAME_SPACE_URI, ELEMENT_OUT);
                out.setTextContent(dval);

                Element origin = document.createElementNS(NAME_SPACE_URI, ELEMENT_ORIGIN);
                origin.setTextContent(cval);

                inlineTable.withRows(new Row().withContent(origin).withContent(out));
            }
        }

        MapValues mapValues = new MapValues("out")
                .withDataType(DataType.DOUBLE)
                .withDefaultValue(defaultValue)
                .withFieldColumnPairs(
                        new FieldColumnPair(new FieldName(CommonUtils.getSimpleColumnName(config, columnConfigList,
                                segmentExpansions, datasetHeaders)), ELEMENT_ORIGIN)).withInlineTable(inlineTable)
                .withMapMissingTo(missingValue);
        List<DerivedField> derivedFields = new ArrayList<DerivedField>();
        derivedFields.add(new DerivedField(OpType.CONTINUOUS, DataType.DOUBLE).withName(
                FieldName.create(genPmmlColumnName(CommonUtils.getSimpleColumnName(config.getColumnName()), normType)))
                .withExpression(mapValues));
        return derivedFields;
    }

    /**
     * Create @DerivedField for numerical variable
     * 
     * @param config
     *            - ColumnConfig for numerical variable
     * @param cutoff
     *            - cutoff of normalization
     * @param normType
     *            - the normalization method that is used to generate DerivedField
     * @return DerivedField for variable
     */
    protected List<DerivedField> createNumericalDerivedField(ColumnConfig config, double cutoff,
            ModelNormalizeConf.NormType normType) {
        // added capping logic to linearNorm
        LinearNorm from = new LinearNorm().withOrig(config.getMean() - config.getStdDev() * cutoff).withNorm(-cutoff);
        LinearNorm to = new LinearNorm().withOrig(config.getMean() + config.getStdDev() * cutoff).withNorm(cutoff);
        NormContinuous normContinuous = new NormContinuous(FieldName.create(CommonUtils.getSimpleColumnName(config,
                columnConfigList, segmentExpansions, datasetHeaders))).withLinearNorms(from, to).withMapMissingTo(0.0)
                .withOutliers(OutlierTreatmentMethodType.AS_EXTREME_VALUES);

        // derived field name is consisted of FieldName and "_zscl"
        List<DerivedField> derivedFields = new ArrayList<DerivedField>();
        derivedFields.add(new DerivedField(OpType.CONTINUOUS, DataType.DOUBLE).withName(
                FieldName.create(genPmmlColumnName(CommonUtils.getSimpleColumnName(config.getColumnName()), normType)))
                .withExpression(normContinuous));
        return derivedFields;
    }

    /**
     * Convert column name into PMML format(with normalization)
     * 
     * @param columnName
     *            the column name
     * @param normType
     *            the norm type
     * @return - PMML standard column name
     */
    public static String genPmmlColumnName(String columnName, ModelNormalizeConf.NormType normType) {
        return columnName + "_" + normType.name().toLowerCase();
    }

}
