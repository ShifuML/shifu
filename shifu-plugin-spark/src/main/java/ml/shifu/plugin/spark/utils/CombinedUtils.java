/**
 * Copyright [2012-2014] eBay Software Foundation
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
package ml.shifu.plugin.spark.utils;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.xml.transform.sax.SAXSource;
import javax.xml.transform.stream.StreamResult;

import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.jpmml.model.ImportFilter;
import org.jpmml.model.JAXBUtil;
import org.xml.sax.InputSource;

import com.google.common.base.Joiner;
import com.google.common.base.Splitter;

/** 
 * Contains all utils functions used by shifu-spark-plugin.
 */
public class CombinedUtils {

	/**
	 * Creates a list of active fields
	 * @param pmml A PMML object
	 * @param params Params object which contains the modelname
	 * @return activeFields
	 */
    public static List<DerivedField> getActiveFields(PMML pmml, Params params) {
        Model model = PMMLUtils.getModelByName(pmml, params.get("modelName")
                .toString());
        Map<FieldUsageType, List<DerivedField>> fieldMap = PMMLUtils
                .getDerivedFieldsByUsageType(pmml, model);
        List<DerivedField> activeFields = fieldMap.get(FieldUsageType.ACTIVE);
        return activeFields;
    }

    /**
     * Creates list of target fields
     * @param pmml PMML object
     * @param params Params object which contains modelname
     * @return targetFields
     */
    public static List<DerivedField> getTargetFields(PMML pmml, Params params) {
        Model model = PMMLUtils.getModelByName(pmml, params.get("modelName")
                .toString());
        Map<FieldUsageType, List<DerivedField>> fieldMap = PMMLUtils
                .getDerivedFieldsByUsageType(pmml, model);
        List<DerivedField> targetFields = fieldMap.get(FieldUsageType.TARGET);
        return targetFields;
    }

    /**
     * Writes the transformed header 
     * @param pathOutputActiveHeader
     * @param activeFields
     * @param targetFields
     */
    public static void writeTransformationHeader(String pathOutputActiveHeader,
            List<DerivedField> activeFields, List<DerivedField> targetFields) {
        PrintWriter headerWriter = null;
        try {
            headerWriter = new PrintWriter(pathOutputActiveHeader);
            List<String> header = new ArrayList<String>();
            for (DerivedField derivedField : targetFields) {
                header.add("TARGET::" + derivedField.getName().getValue());
            }
            for (DerivedField derivedField : activeFields) {
                header.add("ACTIVE::" + derivedField.getName().getValue());
            }
            headerWriter.print(Joiner.on(",").join(header));
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (headerWriter != null)
                headerWriter.close();
        }
    }

    /**
     * Loads a PMML file given the path and the filesystem
     * @param pathPMML String path
     * @param fs FileSystem object
     * @return pmml PMML object
     * @throws IOException
     */
    public static PMML loadPMML(String pathPMML, FileSystem fs)
            throws IOException {
        // load PMML from any fs- local or hdfs
        InputStream pmmlInputStream = null;
        PMML pmml = null;
        try {
            pmmlInputStream = fs.open(new Path(pathPMML));
            InputSource source = new InputSource(pmmlInputStream);
            SAXSource transformedSource = ImportFilter.apply(source);
            pmml = JAXBUtil.unmarshalPMML(transformedSource);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (pmmlInputStream != null)
                pmmlInputStream.close();
        }
        return pmml;
    }

    /**
     * Saves a PMML to a given filesystem
     * @param pmml PMML object
     * @param pathPMML String path to output file
     * @param fs FileSystem object
     * @throws IOException
     */
    public static void savePMML(PMML pmml, String pathPMML, FileSystem fs) throws IOException {
        
        OutputStream pmmlOutputStream= null;
        try {
            pmmlOutputStream= fs.create(new Path(pathPMML), true);
            StreamResult result = new StreamResult(pmmlOutputStream);
            JAXBUtil.marshalPMML(pmml, result);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if(pmmlOutputStream != null)
                pmmlOutputStream.close();
        }
        
    }
    
    
    /**
     * Creates a map from dataFields and parsedInput containing datafield name to value mapping
     * @param dataFields List of DataField
     * @param parsedInput List containing parsed input
     * @return rawDataMap
     */
    public static Map<String, Object> createDataMap(List<DataField> dataFields,
            List<Object> parsedInput) {
        Map<String, Object> rawDataMap = new HashMap<String, Object>();
        for (int i = 0; i < parsedInput.size(); i++) {
            rawDataMap.put(dataFields.get(i).getName().getValue(),
                    parsedInput.get(i));
        }
        return rawDataMap;
    }

    /**
     * Parses input into list of objects using delimiter
     * @param input input string
     * @param delimiter delimiter String
     * @return parsedInput List of Objects
     */
    public static List<Object> getParsedObjects(String input, String delimiter) {
        List<Object> parsedInput = new ArrayList<Object>();

        // Put this step into shifu.core.utils
        for (String s : Splitter.on(delimiter).split(input)) {
            parsedInput.add(s);
        }
        return parsedInput;
    }

    /**
     * Parses input string using delimiter, and creates datamap containing datafield name to value mapping
     * @param dataFields List of DataField
     * @param input input String
     * @param delimiter delimiter String
     * @return rawDataMap
     */
    public static Map<String, Object> createDataMap(List<DataField> dataFields,
            String input, String delimiter) {
        Map<String, Object> rawDataMap = new HashMap<String, Object>();
        int index = 0;
        for (String s : Splitter.on(delimiter).split(input)) {
            rawDataMap.put(dataFields.get(index).getName().getValue(), s);
            index++;
        }
        return rawDataMap;
    }

}
