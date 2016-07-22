package ml.shifu.shifu.core.pmml.builder.impl;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractPmmlElementCreator;
import org.apache.commons.lang.StringUtils;
import org.dmg.pmml.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.util.ArrayList;
import java.util.List;

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
    public LocalTransformations build() {
        LocalTransformations localTransformations = new LocalTransformations();
        for (ColumnConfig config : columnConfigList) {
            if (config.isFinalSelect()) {
                double cutoff = modelConfig.getNormalizeStdDevCutOff();
                localTransformations.withDerivedFields(config.isCategorical() ?
                        createCategoricalDerivedField(config, cutoff, modelConfig.getNormalizeType())
                            : createNumericalDerivedField(config, cutoff, modelConfig.getNormalizeType()));
            }
        }
        return localTransformations;
    }

    /**
     * Create @DerivedField for categorical variable
     *
     * @param config - ColumnConfig for categorical variable
     * @param cutoff - cutoff for normalization
     * @param normType - the normalization method that is used to generate DerivedField
     * @return DerivedField for variable
     */
    protected List<DerivedField> createCategoricalDerivedField(ColumnConfig config, double cutoff, ModelNormalizeConf.NormType normType) {
        Document document = null;
        try {
            document = DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();
        } catch (ParserConfigurationException e) {
            LOG.error("Fail to create document node.", e);
            throw new RuntimeException("Fail to create document node.", e);
        }

        String defaultValue = Normalizer.normalize(config, "doesn't exist at all...by paypal", cutoff, normType).toString();
        String missingValue = Normalizer.normalize(config, null, cutoff, normType).toString();

        InlineTable inlineTable = new InlineTable();
        for (int i = 0; i < config.getBinCategory().size(); i++) {
            String cval = config.getBinCategory().get(i);
            String dval = Normalizer.normalize(config, cval, cutoff, normType).toString();

            Element out = document.createElementNS(NAME_SPACE_URI, ELEMENT_OUT);
            out.setTextContent(dval);

            Element origin = document.createElementNS(NAME_SPACE_URI, ELEMENT_ORIGIN);
            origin.setTextContent(cval);

            inlineTable.withRows(new Row().withContent(origin).withContent(out));
        }

        MapValues mapValues = new MapValues("out")
                .withDataType(DataType.DOUBLE)
                .withDefaultValue(defaultValue)
                .withFieldColumnPairs(new FieldColumnPair(new FieldName(config.getColumnName()), ELEMENT_ORIGIN))
                .withInlineTable(inlineTable)
                .withMapMissingTo(missingValue);

        List<DerivedField> derivedFields = new ArrayList<DerivedField>();
        derivedFields.add(new DerivedField(OpType.CONTINUOUS, DataType.DOUBLE).withName(
                FieldName.create(genPmmlColumnName(config.getColumnName(), normType))).withExpression(mapValues));
        return derivedFields;
    }

    /**
     * Create @DerivedField for numerical variable
     *
     * @param config - ColumnConfig for numerical variable
     * @param cutoff - cutoff of normalization
     * @param normType - the normalization method that is used to generate DerivedField
     * @return DerivedField for variable
     */
    protected List<DerivedField> createNumericalDerivedField(ColumnConfig config, double cutoff, ModelNormalizeConf.NormType normType) {
        // added capping logic to linearNorm
        LinearNorm from = new LinearNorm().withOrig(config.getMean() - config.getStdDev() * cutoff).withNorm(-cutoff);
        LinearNorm to = new LinearNorm().withOrig(config.getMean() + config.getStdDev() * cutoff).withNorm(cutoff);
        NormContinuous normContinuous = new NormContinuous(FieldName.create(config.getColumnName()))
                .withLinearNorms(from, to).withMapMissingTo(0.0)
                .withOutliers(OutlierTreatmentMethodType.AS_EXTREME_VALUES);

        // derived field name is consisted of FieldName and "_zscl"
        List<DerivedField> derivedFields = new ArrayList<DerivedField>();
        derivedFields.add(new DerivedField(OpType.CONTINUOUS, DataType.DOUBLE).withName(
                FieldName.create(genPmmlColumnName(config.getColumnName(), normType))).withExpression(normContinuous));
        return derivedFields;
    }

    /**
     * Convert column name into PMML format(with normalization)
     *
     * @param columnName
     * @parm normType
     * @return - PMML standard column name
     */
    protected String genPmmlColumnName(String columnName, ModelNormalizeConf.NormType normType) {
        return columnName + "_" + normType.name().toLowerCase();
    }

}
