/* 
 * Contains all utils functions used by spark normalizer.
 * TODO: These must be integrated into utils of shifu-core.
 * 
 */
package ml.shifu.plugin.spark.norm;

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

public class CombinedUtils {

    // take PMML object and modelname (from params) and create list of active
    // fields
    public static List<DerivedField> getActiveFields(PMML pmml, Params params) {
        Model model = PMMLUtils.getModelByName(pmml, params.get("modelName")
                .toString());
        Map<FieldUsageType, List<DerivedField>> fieldMap = PMMLUtils
                .getDerivedFieldsByUsageType(pmml, model);
        List<DerivedField> activeFields = fieldMap.get(FieldUsageType.ACTIVE);
        return activeFields;
    }

    // take PMML object and modelname (from params) and create list of target
    // fields
    public static List<DerivedField> getTargetFields(PMML pmml, Params params) {
        Model model = PMMLUtils.getModelByName(pmml, params.get("modelName")
                .toString());
        Map<FieldUsageType, List<DerivedField>> fieldMap = PMMLUtils
                .getDerivedFieldsByUsageType(pmml, model);
        List<DerivedField> targetFields = fieldMap.get(FieldUsageType.TARGET);
        return targetFields;
    }

    // TODO: include functionality for having HDFS pathOutputActiveHeaders
    // write output file header to local FS
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

    // Should go into PMMLUtils
    // load PMML from any filesystem given the path string and filesystem
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
    
    
    // create data map from list of datafields and objects, suitable to be
    // passed
    // as input to TransformExecutor.transform()
    public static Map<String, Object> createDataMap(List<DataField> dataFields,
            List<Object> parsedInput) {
        Map<String, Object> rawDataMap = new HashMap<String, Object>();
        for (int i = 0; i < parsedInput.size(); i++) {
            rawDataMap.put(dataFields.get(i).getName().getValue(),
                    parsedInput.get(i));
        }
        return rawDataMap;
    }

    // take string and delimiter and parse into list of objects
    public static List<Object> getParsedObjects(String input, String delimiter) {
        List<Object> parsedInput = new ArrayList<Object>();

        // Put this step into shifu.core.utils
        for (String s : Splitter.on(delimiter).split(input)) {
            parsedInput.add(s);
        }
        return parsedInput;
    }

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
