package ml.shifu.core.util;

import org.apache.commons.io.IOUtils;
import org.dmg.pmml.*;
import org.jpmml.model.ImportFilter;
import org.jpmml.model.JAXBUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.xml.sax.InputSource;

import javax.xml.transform.sax.SAXSource;
import javax.xml.transform.stream.StreamResult;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.*;


public class PMMLUtils {

    private static Logger log = LoggerFactory.getLogger(PMMLUtils.class);

    private PMMLUtils() {

    }

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

    public static Extension getExtension(List<Extension> extensions, String key) {
        for (Extension extension : extensions) {
            if (key.equals(extension.getName())) {
                return extension;
            }
        }

        throw new RuntimeException("No such extension: " + key);
    }

    public static UnivariateStats getUnivariateStatsByFieldName(ModelStats modelStats, FieldName fieldName) {
        for (UnivariateStats univariateStats : modelStats.getUnivariateStats()) {
            if (univariateStats.getField().equals(fieldName)) {
                return univariateStats;
            }
        }

        throw new RuntimeException("No UnivariateStats for field: " + fieldName);

    }


    public static void savePMML(PMML pmml, String path) {
        OutputStream os = null;

        try {
            os = new FileOutputStream(path);
            StreamResult result = new StreamResult(os);
            JAXBUtil.marshalPMML(pmml, result);
        } catch (Exception e) {
            log.error(e.toString());
        } finally {
            IOUtils.closeQuietly(os);
        }
    }

    public static PMML loadPMML(String path) throws Exception {
        InputStream is = null;

        try {
            is = new FileInputStream(path);
            InputSource source = new InputSource(is);
            SAXSource transformedSource = ImportFilter.apply(source);
            return JAXBUtil.unmarshalPMML(transformedSource);

        } catch (Exception e) {
            log.error(e.toString());
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

            Expression expression = derivedField.getExpression();
            // TODO:finish the list
            if (expression instanceof NormContinuous) {
                FieldName rawFieldName = ((NormContinuous) expression).getField();
                derivedFieldMap.put(rawFieldName, derivedField);
            }


        }

        return derivedFieldMap;
    }


    public static Map<FieldUsageType, List<DerivedField>> getDerivedFieldsByUsageType(PMML pmml, Model model) {

        Map<FieldUsageType, List<DerivedField>> result = new LinkedHashMap<FieldUsageType, List<DerivedField>>();

        List<DerivedField> activeFields = new ArrayList<DerivedField>();
        List<DerivedField> targetFields = new ArrayList<DerivedField>();

        for (DerivedField derivedField : model.getLocalTransformations().getDerivedFields()) {
            Expression expression = derivedField.getExpression();

            List<MiningField> miningFields = model.getMiningSchema().getMiningFields();

            if (expression instanceof NormContinuous) {
                for (MiningField miningField : miningFields) {
                    if (miningField.getName().equals(((NormContinuous) expression).getField())) {

                        if (PMMLUtils.isActive(miningField)) {
                            activeFields.add(derivedField);
                        } else if (PMMLUtils.isTarget(miningField)) {
                            targetFields.add(derivedField);
                        }

                    }
                }
            } else if (expression instanceof MapValues) {
                int active = 0;
                int target = 0;
                List<FieldColumnPair> pairs = ((MapValues) expression).getFieldColumnPairs();
                for (FieldColumnPair pair : pairs) {
                    MiningField miningField = PMMLUtils.getMiningFieldByName(miningFields, pair.getField());
                    if (miningField != null) {
                        if (PMMLUtils.isActive(miningField)) {
                            active += 1;
                        } else if (PMMLUtils.isTarget(miningField)) {
                            target += 1;
                        }
                    }
                }
                if (active == pairs.size()) {
                    activeFields.add(derivedField);
                } else if (target == pairs.size()) {
                    targetFields.add(derivedField);
                }

            } else if (expression instanceof FieldRef) {
                MiningField miningField = PMMLUtils.getMiningFieldByName(miningFields, ((FieldRef) expression).getField());
                if (miningField != null) {
                    if (PMMLUtils.isActive(miningField)) {
                        activeFields.add(derivedField);
                    } else if (PMMLUtils.isTarget(miningField)) {
                        targetFields.add(derivedField);
                    }
                }
            }
        }

        result.put(FieldUsageType.ACTIVE, activeFields);
        result.put(FieldUsageType.TARGET, targetFields);

        return result;
    }

    public static Boolean isActive(MiningField miningField) {
        return miningField.getUsageType().equals(FieldUsageType.ACTIVE);
    }

    public static Boolean isTarget(MiningField miningField) {
        return miningField.getUsageType().equals(FieldUsageType.TARGET);
    }

    public static Boolean containsField(List<MiningField> fields, FieldName fieldName) {
        for (MiningField miningField : fields) {
            if (miningField.getName().equals(fieldName)) {
                return true;
            }
        }
        return false;
    }

    public static Boolean containsField(List<MiningField> fields, FieldName fieldName, FieldUsageType usageType) {
        for (MiningField miningField : fields) {
            if (miningField.getName().equals(fieldName) && (miningField.getUsageType().equals(usageType))) {
                return true;
            }
        }
        return false;
    }

    public static MiningField getMiningFieldByName(List<MiningField> fields, FieldName fieldName) {
        for (MiningField miningField : fields) {
            if (miningField.getName().equals(fieldName)) {
                return miningField;
            }
        }
        return null;
    }

    public static Map<FieldName, MiningField> getMiningFieldMap(MiningSchema miningSchema) {
        Map<FieldName, MiningField> miningFieldMap = new HashMap<FieldName, MiningField>();

        for (MiningField miningField : miningSchema.getMiningFields()) {
            miningFieldMap.put(miningField.getName(), miningField);
        }

        return miningFieldMap;
    }

    public static Integer getNumActiveMiningFields(MiningSchema miningSchema) {

        Integer cnt = 0;
        for (MiningField miningField : miningSchema.getMiningFields()) {
            if (miningField.getUsageType().equals(FieldUsageType.ACTIVE)) {
                cnt += 1;
            }
        }

        return cnt;
    }

    public static Integer getNumTargetMiningFields(MiningSchema miningSchema) {

        Integer cnt = 0;
        for (MiningField miningField : miningSchema.getMiningFields()) {
            if (miningField.getUsageType().equals(FieldUsageType.TARGET)) {
                cnt += 1;
            }
        }


        return cnt;
    }

    public static NeuralInputs createNeuralInputs(PMML pmml, NeuralNetwork model) {

        Map<FieldUsageType, List<DerivedField>> fieldsMap = PMMLUtils.getDerivedFieldsByUsageType(pmml, model);

        List<DerivedField> activeFields = fieldsMap.get(FieldUsageType.ACTIVE);
        List<DerivedField> targetFields = fieldsMap.get(FieldUsageType.TARGET);


        NeuralInputs neuralInputs = new NeuralInputs();

        int neuralInputIndex = 0;
        for (DerivedField derivedField : activeFields) {
            NeuralInput neuralInput = new NeuralInput();
            DerivedField neuralInputDerivedField = new DerivedField();


            FieldRef fieldRef = new FieldRef();
            fieldRef.setField(derivedField.getName());
            neuralInputDerivedField.setExpression(fieldRef);

            neuralInput.setId("0," + neuralInputIndex);
            neuralInputIndex += 1;
            neuralInput.setDerivedField(neuralInputDerivedField);
            neuralInputs.withNeuralInputs(neuralInput);
        }

        //bias
        NeuralInput biasNeuralInput = new NeuralInput();
        biasNeuralInput.setId("bias");

        DerivedField derivedField = new DerivedField();
        derivedField.setExpression(new Constant("1.0"));
        biasNeuralInput.setDerivedField(derivedField);
        neuralInputs.withNeuralInputs(biasNeuralInput);
        neuralInputs.setNumberOfInputs(neuralInputs.getNeuralInputs().size());
        return neuralInputs;
    }


}
