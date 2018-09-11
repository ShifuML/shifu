package ml.shifu.shifu.core.pmml.builder.impl;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf;
import ml.shifu.shifu.util.CommonUtils;
import org.dmg.pmml.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.util.ArrayList;
import java.util.List;

public class ZscoreOneHotLocalTransformCreator extends ZscoreLocalTransformCreator {

    private static final Logger LOG = LoggerFactory.getLogger(ZscoreOneHotLocalTransformCreator.class);

    public ZscoreOneHotLocalTransformCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        super(modelConfig, columnConfigList);
    }

    public ZscoreOneHotLocalTransformCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, boolean isConcise) {
        super(modelConfig, columnConfigList, isConcise);
    }

    @Override
    protected List<DerivedField> createCategoricalDerivedField(ColumnConfig config, double cutoff,
                                                               ModelNormalizeConf.NormType normType) {
        List<DerivedField> derivedFields = new ArrayList<DerivedField>();

        List<String> allValidCatVals = new ArrayList<String>();

        for (int i = 0; i < config.getBinCategory().size(); i++) {
            String category = config.getBinCategory().get(i);
            List<String> catUnits = CommonUtils.flattenCatValGrp(category);
            derivedFields.add(createOneHotDerivedFields(config, i, catUnits, normType, "1.0", "0.0"));
            allValidCatVals.addAll(catUnits);
        }

        // add missing bin
        derivedFields.add(createOneHotDerivedFields(config, config.getBinCategory().size(),
                allValidCatVals, normType, "0.0", "1.0"));

        return derivedFields;
    }

    private DerivedField createOneHotDerivedFields(ColumnConfig config, int ops, List<String> catUnits,
                                                   ModelNormalizeConf.NormType normType,
                                                   String matchVal, String missDef) {
        Document document = null;
        try {
            document = DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();
        } catch (ParserConfigurationException e) {
            LOG.error("Fail to create document node.", e);
            throw new RuntimeException("Fail to create document node.", e);
        }

        String defaultValue = missDef;
        String missingValue = missDef;

        InlineTable inlineTable = new InlineTable();
        for (String cval : catUnits) {
            Element out = document.createElementNS(NAME_SPACE_URI, ELEMENT_OUT);
            out.setTextContent(matchVal);

            Element origin = document.createElementNS(NAME_SPACE_URI, ELEMENT_ORIGIN);
            origin.setTextContent(cval);
            inlineTable.addRows(new Row().addContent(origin).addContent(out));
        }

        MapValues mapValues = new MapValues("out")
                .setDataType(DataType.DOUBLE)
                .setDefaultValue(defaultValue)
                .addFieldColumnPairs(new FieldColumnPair(new FieldName(
                        CommonUtils.getSimpleColumnName(config, columnConfigList, segmentExpansions, datasetHeaders)),
                        ELEMENT_ORIGIN)).setInlineTable(inlineTable)
                .setMapMissingTo(missingValue);

        return new DerivedField(OpType.CONTINUOUS, DataType.DOUBLE).setName(FieldName.create(genPmmlColumnName(
                CommonUtils.getSimpleColumnName(config.getColumnName()), normType) + "_" + ops))
                .setExpression(mapValues);
    }

}
