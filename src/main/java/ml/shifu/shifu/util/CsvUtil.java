package ml.shifu.shifu.util;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningField;
import org.jpmml.evaluator.EvaluationException;
import org.jpmml.evaluator.Evaluator;
import org.jpmml.evaluator.EvaluatorUtil;
import org.jpmml.evaluator.FieldValue;


public class CsvUtil {
	private CsvUtil(){
	}

	static
	public Table readTable(File file) throws IOException {
		return readTable(file, null);
	}

	static
	public Table readTable(File file, String separator) throws IOException {
		Table table = new Table();

		BufferedReader reader = new BufferedReader(new FileReader(file));

		try {
			while(true){
				String line = reader.readLine();
				if(line == null){
					break;
				} // End if

				if((line.trim()).equals("")){
					break;
				} // End if

				if(separator == null){
					separator = getSeparator(line);
				}

				table.add(parseLine(line, separator));
			}
		} finally {
			reader.close();
		}

		table.setSeparator(separator);

		return table;
	}

	static
	public void writeTable(Table table, File file) throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(file));

		try {
			String terminator = "";

			for(List<String> row : table){
				StringBuilder sb = new StringBuilder();

				sb.append(terminator);
				terminator = "\n";

				String separator = "";

				for(int i = 0; i < row.size(); i++){
					sb.append(separator);
					separator = table.getSeparator();

					sb.append(row.get(i));
				}

				writer.write(sb.toString());
			}

			writer.flush();
		} finally {
			writer.close();
		}
	}

	static
	private String getSeparator(String line){
		String[] separators = {"\t", ";", ","};

		for(String separator : separators){
			String[] cells = line.split(separator);

			if(cells.length > 1){
				return separator;
			}
		}

		throw new IllegalArgumentException();
	}

	static
	public List<String> parseLine(String line, String separator){
		List<String> result = new ArrayList<String>();

		String[] cells = line.split(separator);
		for(String cell : cells){

			// Remove quotation marks, if any
			cell = stripQuotes(cell, "\"");
			cell = stripQuotes(cell, "\'");

			// Standardize decimal marks to Full Stop (US)
			if(!(",").equals(separator)){
				cell = cell.replace(',', '.');
			}

			result.add(cell);
		}

		return result;
	}

	static
	private String stripQuotes(String string, String quote){

		if(string.startsWith(quote) && string.endsWith(quote)){
			string = string.substring(quote.length(), string.length() - quote.length());
		}

		return string;
	}
	
	static
	public List<Map<FieldName, FieldValue>> prepareAll(Evaluator evaluator, Table table){
		List<FieldName> names = new ArrayList<FieldName>();

		List<FieldName> activeFields = evaluator.getActiveFields();
		List<FieldName> groupFields = evaluator.getGroupFields();

		header:
		{
			List<String> headerRow = table.get(0);
			for(int column = 0; column < headerRow.size(); column++){
				FieldName field = FieldName.create(headerRow.get(column));

				if(!(activeFields.contains(field) || groupFields.contains(field))){
					field = null;
				}

				names.add(field);
			}
		}

		List<Map<FieldName, Object>> stringRows = new ArrayList<Map<FieldName, Object>>();

		body:
		for(int row = 1; row < table.size(); row++){
			List<String> bodyRow = table.get(row);

			Map<FieldName, Object> stringRow = new LinkedHashMap<FieldName, Object>();

			for(int column = 0; column < bodyRow.size(); column++){
				FieldName name = names.get(column);
				if(name == null){
					continue;
				}

				String value = bodyRow.get(column);
				if(("").equals(value) || ("NA").equals(value) || ("N/A").equals(value)){
					value = null;
				}

				stringRow.put(name, value);
			}

			stringRows.add(stringRow);
		}

		if(groupFields.size() == 1){
			FieldName groupField = groupFields.get(0);

			stringRows = EvaluatorUtil.groupRows(groupField, stringRows);
		} else

		if(groupFields.size() > 1){
			throw new EvaluationException();
		}

		List<Map<FieldName, FieldValue>> fieldValueRows = new ArrayList<Map<FieldName, FieldValue>>();

		for(Map<FieldName, Object> stringRow : stringRows){
			Map<FieldName, FieldValue> fieldValueRow = new LinkedHashMap<FieldName, FieldValue>();

			Collection<Map.Entry<FieldName, Object>> entries = stringRow.entrySet();
			for(Map.Entry<FieldName, Object> entry : entries){
				FieldName name = entry.getKey();
                // Pre Data process: for numeric variable convert non-double value to null.
                if(evaluator.getDataField(name).getDataType() == DataType.DOUBLE) {
                    try{
                        Double.parseDouble((String) entry.getValue());
                    }
                    catch (Exception e)
                    {
                        entry.setValue(null);
                    }
                }
				FieldValue value = EvaluatorUtil.prepare(evaluator, name, entry.getValue());
				fieldValueRow.put(name, value);
			}

			fieldValueRows.add(fieldValueRow);
		}

		return fieldValueRows;
	}
	
	static
	public List<Map<FieldName, FieldValue>> load(Evaluator evaluator,String dataPath, String c) throws IOException
	{
		Table table = CsvUtil.readTable(new File(dataPath), c);
		return CsvUtil.prepareAll(evaluator, table);
	}

	static
	public class Table extends ArrayList<List<String>> {

		private String separator = null;


		public String getSeparator(){
			return this.separator;
		}

		public void setSeparator(String separator){
			this.separator = separator;
		}
	}
}
