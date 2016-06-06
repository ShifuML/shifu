package ml.shifu.shifu.core.pmml.builder.impl;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractPmmlElementCreator;
import org.dmg.pmml.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.util.List;

/**
 * Created by zhanhu on 3/29/16.
 */
public class ZscoreLocalTransformCreator extends AbstractPmmlElementCreator<LocalTransformations> {

    private static final Logger LOG = LoggerFactory.getLogger(ZscoreLocalTransformCreator.class);

    private static final String NAME_SPACE_URI = "http://www.dmg.org/PMML-4_2";
    private static final String ELEMENT_OUT = "out";
    private static final String ELEMENT_ORIGIN = "origin";

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
                        createCategoricalDerivedField(config, cutoff) : createNumericalDerivedField(config, cutoff));
            }
        }
        return localTransformations;
    }

    /**
     * Create @DerivedField for categorical variable
     *
     * @param config - ColumnConfig for categorical variable
     * @param cutoff - cutoff for normalization
     * @return DerivedField for variable
     */
    protected DerivedField createCategoricalDerivedField(ColumnConfig config, double cutoff) {
        Document document = null;
        try {
            document = DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();
        } catch (ParserConfigurationException e) {
            LOG.error("Fail to create document node.", e);
            throw new RuntimeException("Fail to create document node.", e);
        }

        String defaultValue = Normalizer.normalize(config, "doesn't exist at all...by paypal", cutoff, this.modelConfig.getNormalizeType()).toString();
        String missingValue = Normalizer.normalize(config, null, cutoff, this.modelConfig.getNormalizeType()).toString();

        InlineTable inlineTable = new InlineTable();
        for (int i = 0; i < config.getBinCategory().size(); i++) {
            String cval = config.getBinCategory().get(i);
            String dval = Normalizer.normalize(config, cval, cutoff, this.modelConfig.getNormalizeType()).toString();

            Element out = document.createElementNS(NAME_SPACE_URI, ELEMENT_OUT);
            out.setTextContent(dval);

            Element origin = document.createElementNS(NAME_SPACE_URI, ELEMENT_ORIGIN);
            origin.setTextContent(cval);

            inlineTable.withRows(new Row().withContent(origin).withContent(out));
        }

        MapValues mapValues = new MapValues("out").withDataType(DataType.DOUBLE).withDefaultValue(defaultValue)
                .withFieldColumnPairs(new FieldColumnPair(new FieldName(config.getColumnName()), ELEMENT_ORIGIN))
                .withInlineTable(inlineTable).withMapMissingTo(missingValue);

        return new DerivedField(OpType.CONTINUOUS, DataType.DOUBLE).withName(
                FieldName.create(genPmmlColumnName(config.getColumnName()))).withExpression(mapValues);
    }

    /**
     * Create @DerivedField for numerical variable
     *
     * @param config - ColumnConfig for numerical variable
     * @param cutoff - cutoff of normalization
     * @return DerivedField for variable
     */
    protected DerivedField createNumericalDerivedField(ColumnConfig config, double cutoff) {
        // added capping logic to linearNorm
        LinearNorm from = new LinearNorm().withOrig(config.getMean() - config.getStdDev() * cutoff).withNorm(-cutoff);
        LinearNorm to = new LinearNorm().withOrig(config.getMean() + config.getStdDev() * cutoff).withNorm(cutoff);
        NormContinuous normContinuous = new NormContinuous(FieldName.create(config.getColumnName()))
                .withLinearNorms(from, to).withMapMissingTo(0.0)
                .withOutliers(OutlierTreatmentMethodType.AS_EXTREME_VALUES);

        // derived field name is consisted of FieldName and "_zscl"
        return new DerivedField(OpType.CONTINUOUS, DataType.DOUBLE).withName(
                FieldName.create(genPmmlColumnName(config.getColumnName()))).withExpression(normContinuous);
    }

    /**
     * Convert column name into PMML format(with normalization)
     *
     * @param columnName
     * @return - PMML standard column name
     */
    protected String genPmmlColumnName(String columnName) {
        return columnName + "_" + this.modelConfig.getNormalizeType().name().toLowerCase();
    }

}
