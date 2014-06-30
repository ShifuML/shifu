package ml.shifu.plugin.encog.adapter;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;
import java.util.Map;

import javax.xml.transform.sax.SAXSource;
import javax.xml.transform.stream.StreamResult;

import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.DataField;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.ModelEvaluator;
import org.jpmml.model.ImportFilter;
import org.jpmml.model.JAXBUtil;
import org.xml.sax.InputSource;

public class BasicAdapterTest {

    public PMML readPMMLFile(String initPmmlPath) {
        try {
            // write PMML
            InputStream is = new FileInputStream(initPmmlPath);
            InputSource source = new InputSource(is);
            // Use SAX filtering to transform PMML schema version 3.X and 4.X
            // documents to PMML schema version 4.2 document
            SAXSource transformedSource = ImportFilter.apply(source);

            PMML pmml = JAXBUtil.unmarshalPMML(transformedSource);
            return pmml;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }


    protected void writePMML(String path,PMML pmml) {

        try {
            // write PMML
            OutputStream os = new FileOutputStream(path);
            StreamResult result = new StreamResult(os);
            JAXBUtil.marshalPMML(pmml, result);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

   
}
