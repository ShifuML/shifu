package ml.shifu.shifu.util;

import org.dmg.pmml.Extension;

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
}
