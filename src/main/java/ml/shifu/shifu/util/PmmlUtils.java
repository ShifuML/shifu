/*
 * Copyright [2013-2015] PayPal Software Foundation
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
package ml.shifu.shifu.util;

import org.dmg.pmml.*;
import org.jpmml.model.ImportFilter;
import org.jpmml.model.JAXBUtil;
import org.xml.sax.InputSource;

import javax.xml.transform.sax.SAXSource;
import javax.xml.transform.stream.StreamResult;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class PmmlUtils {

    public static List<Extension> createExtensions(Map<String, String> extensionMap) {

        List<Extension> extensions = new ArrayList<Extension>();

        for(Map.Entry<String, String> entry: extensionMap.entrySet()) {
            Extension extension = new Extension();
            extension.setName(entry.getKey());
            extension.setValue(entry.getValue());
            extensions.add(extension);
        }

        return extensions;
    }

    public static Extension getExtension(List<Extension> extensions, String key) {
        for(Extension extension: extensions) {
            if(key.equals(extension.getName())) {
                return extension;
            }
        }

        throw new RuntimeException("No such extension: " + key);
    }

    public static UnivariateStats getUnivariateStatsByFieldName(ModelStats modelStats, FieldName fieldName) {
        for(UnivariateStats univariateStats: modelStats.getUnivariateStats()) {
            if(univariateStats.getField().equals(fieldName)) {
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
            e.printStackTrace();
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
            e.printStackTrace();
            throw e;
        }
    }

    public static DataType getDefaultDataTypeByOpType(OpType optype) {
        if(optype.equals(OpType.CONTINUOUS)) {
            return DataType.DOUBLE;
        } else {
            return DataType.STRING;
        }
    }

    public static Model createModelByType(String name) {
        if(name.equalsIgnoreCase("NeuralNetwork")) {
            return new NeuralNetwork();
        } else {
            throw new RuntimeException("Model not supported: " + name);
        }

    }

    public static Model getModelByName(PMML pmml, String name) {
        for(Model model: pmml.getModels()) {
            if(model.getModelName().equals(name)) {
                return model;
            }
        }
        throw new RuntimeException("No such model: " + name);

    }

    public static Integer getTargetFieldNumByName(DataDictionary dataDictionary, String name) {
        int size = dataDictionary.getNumberOfFields();
        for(int i = 0; i < size; i++) {
            DataField dataField = dataDictionary.getDataFields().get(i);
            if(dataField.getName().getValue().equals(name)) {
                return i;
            }
        }
        throw new RuntimeException("Target Field Not Found: " + name);

    }

    public static Map<FieldName, Integer> getFieldNumMap(DataDictionary dataDictionary) {
        Map<FieldName, Integer> fieldNumMap = new HashMap<FieldName, Integer>();
        int size = dataDictionary.getNumberOfFields();

        for(int i = 0; i < size; i++) {
            DataField dataField = dataDictionary.getDataFields().get(i);
            fieldNumMap.put(dataField.getName(), i);
        }
        return fieldNumMap;
    }

    public static Map<FieldName, DerivedField> getDerivedFieldMap(LocalTransformations localTransformations) {

        Map<FieldName, DerivedField> derivedFieldMap = new HashMap<FieldName, DerivedField>();

        for(DerivedField derivedField: localTransformations.getDerivedFields()) {
            derivedFieldMap.put(derivedField.getName(), derivedField);
        }

        return derivedFieldMap;
    }

    public static Map<FieldName, MiningField> getMiningFieldMap(MiningSchema miningSchema) {
        Map<FieldName, MiningField> miningFieldMap = new HashMap<FieldName, MiningField>();

        for(MiningField miningField: miningSchema.getMiningFields()) {
            miningFieldMap.put(miningField.getName(), miningField);
        }

        return miningFieldMap;
    }

    public static Integer getNumActiveMiningFields(MiningSchema miningSchema) {

        Integer cnt = 0;
        for(MiningField miningField: miningSchema.getMiningFields()) {
            if(miningField.getUsageType().equals(FieldUsageType.ACTIVE)) {
                cnt += 1;
            }
        }

        return cnt;
    }

    public static Integer getNumTargetMiningFields(MiningSchema miningSchema) {

        Integer cnt = 0;
        for(MiningField miningField: miningSchema.getMiningFields()) {
            if(miningField.getUsageType().equals(FieldUsageType.TARGET)) {
                cnt += 1;
            }
        }

        return cnt;
    }

}
