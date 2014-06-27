package ml.shifu.shifu.di.builtin.derivedField;

import ml.shifu.shifu.container.RawTarget;
import ml.shifu.shifu.di.spi.DerivedFieldCreator;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.PMMLUtils;
import ml.shifu.shifu.util.Params;
import org.dmg.pmml.*;

import java.util.List;

public class BinaryClassMapperCreator implements DerivedFieldCreator {

    public DerivedField create(DataField dataField, ModelStats modelStats, Params params) {


        List<String> posTags = (List<String>) params.get("posTags");
        List<String> negTags = (List<String>) params.get("negTags");

        DerivedField derivedField = new DerivedField();
        derivedField.setName(dataField.getName());
        derivedField.setOptype(dataField.getOptype());
        derivedField.setDataType(dataField.getDataType());

        MapValues mapValues = new MapValues();

        FieldColumnPair fieldColumnPair = new FieldColumnPair();
        fieldColumnPair.setField(dataField.getName());
        fieldColumnPair.setColumn("0");

        mapValues.setOutputColumn("1");
        mapValues.withFieldColumnPairs(fieldColumnPair);

        InlineTable inlineTable = new InlineTable();



        for (String posTag : posTags) {
            Row row = new Row();
            row.withContent(posTag, "1");
            inlineTable.withRows(row);
        }

        for (String negTag : negTags) {
            Row row = new Row();

            row.withContent(negTag, "0");
            inlineTable.withRows(row);
        }



        mapValues.withInlineTable(inlineTable);

        derivedField.setExpression(mapValues);

        return derivedField;
    }
}


