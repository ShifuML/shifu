package ml.shifu.shifu.util;

import ml.shifu.shifu.container.BinningObject;
import org.apache.pig.Expression;
import org.dmg.pmml.*;
import org.jpmml.model.ImportFilter;
import org.jpmml.model.JAXBUtil;
import org.xml.sax.InputSource;

import javax.xml.transform.sax.SAXSource;
import javax.xml.transform.stream.StreamResult;
import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class PMMLUtils {


    public static List<Extension> createExtensions(Map<String, String> extensionMap) {

        List<Extension> extensions = new ArrayList<Extension>();

        for (String key : extensionMap.keySet()) {
            Extension extension = new Extension();
            extension.setName(key);
            extension.setValue(extensionMap.get(key));
            extensions.add(extension);
        }

        return extensions;
    }

    public static void savePMML(PMML pmml, String path) {
        OutputStream os = null;

        try {
            os = new FileOutputStream(path);
            StreamResult result = new StreamResult(os);
            JAXBUtil.marshalPMML(pmml, result);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static PMML loadPMML(String path) throws Exception{
        InputStream is = null;

        try {
            is = new FileInputStream(path);
            InputSource source = new InputSource(is);
            SAXSource transformedSource = ImportFilter.apply(source);
            return JAXBUtil.unmarshalPMML(transformedSource);

        } catch (Exception e) {
            e.printStackTrace();
            throw e;
        }
    }

    public static DataType getDefaultDataTypeByOpType(OpType optype) {
        if (optype.equals(OpType.CONTINUOUS)) {
            return DataType.DOUBLE;
        } else {
            return DataType.STRING;
        }
    }

    public static OpType getOpTypeFromParams(Params params) {
        return OpType.valueOf(params.get("optype").toString().toUpperCase());
    }


    public static DataType getDataTypeFromParams(Params params) {

        if (params.containsKey("dataType")) {
            return DataType.valueOf(params.get("dataType").toString().toUpperCase());
        } else {
            return getDefaultDataTypeByOpType(getOpTypeFromParams(params));
        }
    }

    public static Model createModelByType(String name) {
        if (name.equalsIgnoreCase("NeuralNetwork")) {
            return new NeuralNetwork();
        } else {
            throw new RuntimeException("Model not supported: " + name);
        }

    }

    public static Model getModelByName(PMML pmml, String name) {
        for (Model model : pmml.getModels()) {
            if (model.getModelName().equals(name)) {
                return model;
            }
        }
        throw new RuntimeException("No such model: " + name);

    }

    public static Integer getTargetFieldNumByName(DataDictionary dataDictionary, String name) {
        int size = dataDictionary.getNumberOfFields();
        for (int i = 0; i < size; i++) {
            DataField dataField = dataDictionary.getDataFields().get(i);
            if (dataField.getName().getValue().equals(name)) {
                return i;
            }
        }
        throw new RuntimeException("Target Field Not Found: " + name);

    }

    public static Map<FieldName, Integer> getFieldNumMap(DataDictionary dataDictionary) {
        Map<FieldName, Integer> fieldNumMap = new HashMap<FieldName, Integer>();
        int size = dataDictionary.getNumberOfFields();

        for (int i = 0; i < size; i++) {
            DataField dataField = dataDictionary.getDataFields().get(i);
            fieldNumMap.put(dataField.getName(), i);
        }
        return fieldNumMap;
    }

    public static Map<FieldName, DerivedField> getDerivedFieldMap(LocalTransformations localTransformations) {

        Map<FieldName, DerivedField> derivedFieldMap = new HashMap<FieldName, DerivedField>();

        for (DerivedField derivedField : localTransformations.getDerivedFields()) {
            derivedFieldMap.put(derivedField.getName(), derivedField);
        }

        return derivedFieldMap;
    }

}
