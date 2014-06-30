package ml.shifu.plugin.encog.adapter;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ml.shifu.plugin.PMMLAdapterCommonUtil;

import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Discretize;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.Model;
import org.dmg.pmml.NormContinuous;
import org.dmg.pmml.PMML;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.jpmml.evaluator.DiscretizationUtil;
import org.jpmml.evaluator.ModelEvaluationContext;
import org.jpmml.evaluator.ModelEvaluator;
import org.jpmml.evaluator.NormalizationUtil;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

public class EvalCSVUtil {
    private String path;

    private List<Map<FieldName, String>> table = Lists.newArrayList();
    private PMML pmml;
    // private List<double[]> dataSet = new ArrayList<double[]>();
    private BufferedReader reader;
    private String[] headers;

    public EvalCSVUtil(String dataPath, PMML pmml) {
        this.path = dataPath;
        this.pmml = pmml;
        headers = PMMLAdapterCommonUtil.getDataDicHeaders(pmml);
        parseCSV();

    }

    private void parseCSV() {
        try {
            reader = new BufferedReader(new FileReader(path));
            List<FieldName> keys = Lists.newArrayList();

            for (int i = 0; i < headers.length; i++) {
                keys.add(FieldName.create(headers[i]));
            }
            while (true) {
                String bodyLine = reader.readLine();
                if (bodyLine == null) {
                    break;
                }

                Map<FieldName, String> row = Maps.newLinkedHashMap();

                String[] bodyCells = bodyLine.split(",");
                // dataSet.add(convertToDouble(bodyCells));
                // Must be of equal length
                if (bodyCells.length != headers.length) {
                    throw new RuntimeException();
                }

                for (int i = 0; i < bodyCells.length; i++) {
                    row.put(keys.get(i), bodyCells[i]);
                }

                table.add(row);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    /**
     * Return a list of fieldName-value map that can be used as the input of
     * PMML evaluator.
     * 
     * @returna The list of fieldName-value map that is used as the input of
     *          PMML evaluator.
     */
    public List<Map<FieldName, String>> getEvaluatorInput() {
        return table;
    }

    private double[] convertToDouble(String[] tokens) {
        int len = tokens.length;
        double[] itemList = new double[len];
        for (int i = 0; i < len; i++)
            itemList[i] = Double.parseDouble(tokens[i]);
        return itemList;
    }

    /**
     * <p>
     * Construct the Encog MLDataSet from CSV files, as the input training set
     * for the Encog framework.
     * <p>
     * Notice that, the input of MLDataSet includes a bias, which is always set
     * to 1.
     * 
     * @return The MLDataSet that can be used as the input training data set for
     *         Encog framework.
     */
    public MLData getEncogMLDataSet(List<Double> data) {
      int len = data.size();
         double[] itemList = new double[len+1];
         for (int i = 0; i < len ; i++)
         itemList[i] = data.get(i);
         itemList[len] = 1;
     return  new BasicMLData(itemList);
       
    }

    // public List<MahoutDataPair> getMahoutDataPair() {
    // List<MahoutDataPair> dataPairList = new ArrayList<MahoutDataPair>();
    // for (double[] fields : dataSet) {
    // int len = fields.length;
    // double[] itemList = new double[len-1];
    // for (int i = 0; i < len - 1; i++)
    // itemList[i] = fields[i + 1];
    // dataPairList.add(new MahoutDataPair((int) fields[0], itemList));
    // }
    // return dataPairList;
    // }

    private double transform(DerivedField derivedField, Object origin) {

        Expression expression = derivedField.getExpression();

        // TODO: finish the list
        if (expression instanceof NormContinuous) {
            return NormalizationUtil.normalize((NormContinuous) expression,
                    Double.parseDouble(origin.toString()));
        } else if (expression instanceof Discretize) {
            return   Double.parseDouble(DiscretizationUtil.discretize((Discretize) expression,
                    Double.parseDouble(origin.toString())));
        } else {
            throw new RuntimeException("Invalid Expression(Field: "
                    + derivedField.getName().getValue() + ")");
        }

    }

   
    public MLData normalizeData(ModelEvaluationContext context) {
        Model model = pmml.getModels().get(0);

        List<String> selectedFields = PMMLAdapterCommonUtil
                .getSchemaSelectedFields(model.getMiningSchema());
        List<DerivedField> derivedFields = model.getLocalTransformations()
                .getDerivedFields();

        List<Double> transformed = new ArrayList<Double>();
        for (DerivedField df : derivedFields) {
            if (selectedFields.contains(df.getName().getValue())) {
                transformed.add(transform(df,context.getFields().get(df.getName()).getValue()));
            }
        }
        return getEncogMLDataSet(transformed);
    }
    protected double getPMMLEvaluatorResult(ModelEvaluator evaluator,
            Map<FieldName, String> inputData) {
        if (evaluator == null)
            return 0;
        @SuppressWarnings("unchecked")
        Map<FieldName, Double> evalMap = (Map<FieldName, Double>) evaluator
                .evaluate(inputData);
        for (Map.Entry<FieldName, Double> entry : evalMap.entrySet()) {
            return entry.getValue();
        }
        return 0;
    }
}
