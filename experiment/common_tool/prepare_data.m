function DATA = prepare_data(dataname, source_domain, target_domain, trial)

datapath = ['../data/' dataname];

%% data
load(fullfile(datapath, ['/', source_domain, '.mat']), 'features', 'labels');
source_features = features';
source_labels = labels;
clear features labels

load(fullfile(datapath, ['/', target_domain, '.mat']), 'features', 'labels');
target_features = features';
target_labels = labels;
clear features labels


%% index
load(fullfile(datapath,[source_domain '_' target_domain '_index.mat']), 'source_train_indices', 'target_train_indices', 'target_test_indices');
source_index = source_train_indices{trial};
target_training_index = target_train_indices{trial};
target_test_index = target_test_indices{trial};
clear source_train_indices target_train_indices target_test_indices


source_features = source_features ./ repmat(sqrt(sum(source_features.^2)), size(source_features, 1), 1);
target_features = target_features ./ repmat(sqrt(sum(target_features.^2)), size(target_features, 1), 1);


source_features = source_features(:, source_index);
source_labels   = source_labels(source_index);


target_labeled_features = target_features(:, target_training_index);
target_test_features    = target_features(:, target_test_index);
clear target_features

target_labeled_labels   = target_labels(target_training_index);
target_test_labels      = target_labels(target_test_index);
clear target_labels

% return
DATA.source_features = source_features;
DATA.source_labels = source_labels;

DATA.target_labeled_features = target_labeled_features;
DATA.target_labeled_labels = target_labeled_labels;

DATA.target_unlabeled_features = target_test_features;
DATA.target_unlabeled_labels = target_test_labels;

DATA.num_class = length(unique(source_labels));
