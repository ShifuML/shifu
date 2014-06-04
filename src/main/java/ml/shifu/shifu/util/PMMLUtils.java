package ml.shifu.shifu.util;

import org.dmg.pmml.Extension;
import org.dmg.pmml.PMML;
import org.jpmml.model.ImportFilter;
import org.jpmml.model.JAXBUtil;
import org.xml.sax.InputSource;

import javax.xml.transform.sax.SAXSource;
import javax.xml.transform.stream.StreamResult;
import java.io.*;
import java.util.ArrayList;
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
}
