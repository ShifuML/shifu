package ml.shifu.core.di.builtin.derivedfield;

import ml.shifu.core.di.spi.PMMLDerivedFieldCreator;
import ml.shifu.core.util.Params;
import org.dmg.pmml.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.util.List;

public class BinaryClassMapperCreator implements PMMLDerivedFieldCreator {

    private static final Logger log = LoggerFactory.getLogger(BinaryClassMapperCreator.class);

    public DerivedField create(DataField dataField, ModelStats modelStats, Params params) {


        List<String> posTags = (List<String>) params.get("posTags");
        List<String> negTags = (List<String>) params.get("negTags");

        DerivedField derivedField = new DerivedField();
        derivedField.setName(new FieldName(dataField.getName().getValue() + "_LocalTransformed"));
        derivedField.setOptype(dataField.getOptype());
        derivedField.setDataType(dataField.getDataType());

        MapValues mapValues = new MapValues();

        FieldColumnPair fieldColumnPair = new FieldColumnPair();
        fieldColumnPair.setField(dataField.getName());
        fieldColumnPair.setColumn("origin");
        mapValues.setDataType(derivedField.getDataType());
        mapValues.setOutputColumn("transformed");
        mapValues.withFieldColumnPairs(fieldColumnPair);

        InlineTable inlineTable = new InlineTable();

        DocumentBuilderFactory factory =
                DocumentBuilderFactory.newInstance();

        DocumentBuilder builder;

        try {
            builder = factory.newDocumentBuilder();
        } catch (Exception e) {
            log.error(e.toString());
            return null;
        }
        Document doc = builder.newDocument();

        for (String posTag : posTags) {
            Row row = new Row();


            Element origin = doc.createElement("origin");
            origin.setTextContent(posTag);


            Element transformed = doc.createElement("transformed");
            transformed.setTextContent("1");

            row.withContent(origin, transformed);

            inlineTable.withRows(row);
        }

        for (String negTag : negTags) {
            Row row = new Row();

            Element origin = doc.createElement("origin");
            origin.setTextContent(negTag);


            Element transformed = doc.createElement("transformed");
            transformed.setTextContent("0");

            row.withContent(origin, transformed);
            inlineTable.withRows(row);
        }


        mapValues.withInlineTable(inlineTable);

        derivedField.setExpression(mapValues);

        return derivedField;
    }
}


