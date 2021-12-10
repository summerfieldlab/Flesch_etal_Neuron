function data = load_struct(filePath,fileName)
    % loads and returns struct
    data = load([filePath fileName]);
    fns = fieldnames(data);
    data = data.(fns{1});
end