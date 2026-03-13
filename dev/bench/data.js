window.BENCHMARK_DATA = {
  "lastUpdate": 1773417250577,
  "repoUrl": "https://github.com/xorq-labs/xorq-gallery",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "name": "xorq-labs",
            "username": "xorq-labs"
          },
          "committer": {
            "name": "xorq-labs",
            "username": "xorq-labs"
          },
          "id": "85d7f13f8cf35f90e4db0f2f07f9ebc70fd97d24",
          "message": "Add build artifact caching and catalog sync",
          "timestamp": "2026-03-11T11:15:48Z",
          "url": "https://github.com/xorq-labs/xorq-gallery/pull/3/commits/85d7f13f8cf35f90e4db0f2f07f9ebc70fd97d24"
        },
        "date": 1773359914494,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_list_groups",
            "value": 5364.7621813944015,
            "unit": "iter/sec",
            "range": "stddev: 0.00033021453288001416",
            "extra": "mean: 186.40155261087108 usec\nrounds: 1283"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_all_scripts",
            "value": 4669.7960087921565,
            "unit": "iter/sec",
            "range": "stddev: 0.000361299062776527",
            "extra": "mean: 214.14211629742053 usec\nrounds: 2528"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_cyclical_feature_engineering]",
            "value": 2328.0965106187964,
            "unit": "iter/sec",
            "range": "stddev: 0.0000475486930776509",
            "extra": "mean: 429.5354575890005 usec\nrounds: 1344"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_time_series_lagged_features]",
            "value": 2057.97450668709,
            "unit": "iter/sec",
            "range": "stddev: 0.0007677202376055774",
            "extra": "mean: 485.9146684036391 usec\nrounds: 2111"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_tomography_l1_reconstruction]",
            "value": 2040.0693795887619,
            "unit": "iter/sec",
            "range": "stddev: 0.0010777750905628773",
            "extra": "mean: 490.1794076246468 usec\nrounds: 2046"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_topics_extraction_with_nmf_lda]",
            "value": 2309.837895563966,
            "unit": "iter/sec",
            "range": "stddev: 0.00005107826294422144",
            "extra": "mean: 432.9308138551609 usec\nrounds: 1891"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_compare_calibration]",
            "value": 2302.2794298732397,
            "unit": "iter/sec",
            "range": "stddev: 0.0011561667516904654",
            "extra": "mean: 434.3521411973258 usec\nrounds: 2238"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_classification_probability]",
            "value": 2364.780153812327,
            "unit": "iter/sec",
            "range": "stddev: 0.00004755050526112874",
            "extra": "mean: 422.8722904274516 usec\nrounds: 2152"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_classifier_comparison]",
            "value": 2133.226959483971,
            "unit": "iter/sec",
            "range": "stddev: 0.0017260213540315186",
            "extra": "mean: 468.77337432576826 usec\nrounds: 1854"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_lda_qda]",
            "value": 2352.543034604137,
            "unit": "iter/sec",
            "range": "stddev: 0.00004644723320035439",
            "extra": "mean: 425.0719265453396 usec\nrounds: 1974"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_affinity_propagation]",
            "value": 2342.8776892934034,
            "unit": "iter/sec",
            "range": "stddev: 0.00005505665762147324",
            "extra": "mean: 426.8255251095047 usec\nrounds: 2051"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_kmeans_digits]",
            "value": 2123.02663179036,
            "unit": "iter/sec",
            "range": "stddev: 0.002060974110724872",
            "extra": "mean: 471.02565037382243 usec\nrounds: 2005"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_kmeans_silhouette_analysis]",
            "value": 2360.4433380318337,
            "unit": "iter/sec",
            "range": "stddev: 0.00004723615080191593",
            "extra": "mean: 423.64922889181156 usec\nrounds: 1516"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_column_transformer_mixed_types]",
            "value": 2410.5062916583433,
            "unit": "iter/sec",
            "range": "stddev: 0.00004322724088649245",
            "extra": "mean: 414.85060771695197 usec\nrounds: 1555"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_feature_union]",
            "value": 2112.830474708036,
            "unit": "iter/sec",
            "range": "stddev: 0.002601719433125255",
            "extra": "mean: 473.29873928393914 usec\nrounds: 1983"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_faces_decomposition]",
            "value": 2447.799611706031,
            "unit": "iter/sec",
            "range": "stddev: 0.00004687403392893968",
            "extra": "mean: 408.5301734740593 usec\nrounds: 2179"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_gradient_boosting_categorical]",
            "value": 2374.103218494104,
            "unit": "iter/sec",
            "range": "stddev: 0.00004331401607390398",
            "extra": "mean: 421.2116778285238 usec\nrounds: 1971"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_stack_predictors]",
            "value": 2360.3979157095678,
            "unit": "iter/sec",
            "range": "stddev: 0.0000467498564375252",
            "extra": "mean: 423.6573813866407 usec\nrounds: 2192"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_voting_regressor]",
            "value": 2067.8530200768923,
            "unit": "iter/sec",
            "range": "stddev: 0.0028907027133328374",
            "extra": "mean: 483.5933648527956 usec\nrounds: 2242"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_rfe_digits]",
            "value": 2420.6022990651577,
            "unit": "iter/sec",
            "range": "stddev: 0.000046093364643652466",
            "extra": "mean: 413.12032149444883 usec\nrounds: 2168"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_select_from_model_diabetes]",
            "value": 2396.4725388069505,
            "unit": "iter/sec",
            "range": "stddev: 0.00005027191751237988",
            "extra": "mean: 417.2799745486905 usec\nrounds: 2161"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_lasso_and_elasticnet]",
            "value": 2368.718761013242,
            "unit": "iter/sec",
            "range": "stddev: 0.00004604190732265922",
            "extra": "mean: 422.16915594160304 usec\nrounds: 2129"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_logistic_multinomial]",
            "value": 2355.927701575497,
            "unit": "iter/sec",
            "range": "stddev: 0.000060646505358731806",
            "extra": "mean: 424.4612427330697 usec\nrounds: 1961"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_quantile_regression]",
            "value": 1887.379725792224,
            "unit": "iter/sec",
            "range": "stddev: 0.003942095230421582",
            "extra": "mean: 529.8350863551065 usec\nrounds: 2096"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_confusion_matrix]",
            "value": 2373.9575140186043,
            "unit": "iter/sec",
            "range": "stddev: 0.00004580857657280086",
            "extra": "mean: 421.2375301979238 usec\nrounds: 2020"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_cv_indices]",
            "value": 2384.225523412341,
            "unit": "iter/sec",
            "range": "stddev: 0.00003983682825361853",
            "extra": "mean: 419.4234103193327 usec\nrounds: 1628"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_roc]",
            "value": 2379.539492656718,
            "unit": "iter/sec",
            "range": "stddev: 0.00004405771975172559",
            "extra": "mean: 420.24938148158907 usec\nrounds: 1890"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_nca_classification]",
            "value": 2467.0548976716505,
            "unit": "iter/sec",
            "range": "stddev: 0.00004152603876104085",
            "extra": "mean: 405.341608305424 usec\nrounds: 2119"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_mlp_alpha]",
            "value": 2472.964231854464,
            "unit": "iter/sec",
            "range": "stddev: 0.000041360226164036714",
            "extra": "mean: 404.37301402054845 usec\nrounds: 2211"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_all_scaling]",
            "value": 1891.692655779509,
            "unit": "iter/sec",
            "range": "stddev: 0.004760288114672587",
            "extra": "mean: 528.6270985642382 usec\nrounds: 2019"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_discretization_strategies]",
            "value": 2375.174513089971,
            "unit": "iter/sec",
            "range": "stddev: 0.00004841088126894844",
            "extra": "mean: 421.02169524337614 usec\nrounds: 2018"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_target_encoder]",
            "value": 2392.3301314513315,
            "unit": "iter/sec",
            "range": "stddev: 0.000044740205077624914",
            "extra": "mean: 418.002510127371 usec\nrounds: 2123"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_svm_kernels]",
            "value": 2403.286020127879,
            "unit": "iter/sec",
            "range": "stddev: 0.000046174722680185716",
            "extra": "mean: 416.0969570932677 usec\nrounds: 2051"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_svm_regression]",
            "value": 2404.672199069623,
            "unit": "iter/sec",
            "range": "stddev: 0.000042521068623452456",
            "extra": "mean: 415.8570970242447 usec\nrounds: 1546"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_document_classification_20newsgroups]",
            "value": 2342.5399870757283,
            "unit": "iter/sec",
            "range": "stddev: 0.0000746346528223532",
            "extra": "mean: 426.8870565784167 usec\nrounds: 2227"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_tree_regression]",
            "value": 2489.663796016509,
            "unit": "iter/sec",
            "range": "stddev: 0.00003847158195000239",
            "extra": "mean: 401.66065859977226 usec\nrounds: 1157"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_cyclical_feature_engineering]",
            "value": 2.462106150186793,
            "unit": "iter/sec",
            "range": "stddev: 0.006534893387265886",
            "extra": "mean: 406.1563308000075 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_time_series_lagged_features]",
            "value": 21.16428765092474,
            "unit": "iter/sec",
            "range": "stddev: 0.000730306636806307",
            "extra": "mean: 47.249405058823534 msec\nrounds: 17"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_tomography_l1_reconstruction]",
            "value": 0.15151657658365125,
            "unit": "iter/sec",
            "range": "stddev: 0.5093669057262457",
            "extra": "mean: 6.599937924599999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_topics_extraction_with_nmf_lda]",
            "value": 0.06963697143227415,
            "unit": "iter/sec",
            "range": "stddev: 0.09894641730830403",
            "extra": "mean: 14.360187978199997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_compare_calibration]",
            "value": 4.322774028947418,
            "unit": "iter/sec",
            "range": "stddev: 0.006541591588080539",
            "extra": "mean: 231.33293420000882 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_classification_probability]",
            "value": 16.964168605388473,
            "unit": "iter/sec",
            "range": "stddev: 0.0030836102452555985",
            "extra": "mean: 58.9477753529496 msec\nrounds: 17"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_classifier_comparison]",
            "value": 0.44870159696752515,
            "unit": "iter/sec",
            "range": "stddev: 0.03004128853188743",
            "extra": "mean: 2.2286526430000095 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_lda_qda]",
            "value": 41.86784331901563,
            "unit": "iter/sec",
            "range": "stddev: 0.0007630488864646336",
            "extra": "mean: 23.884679045452952 msec\nrounds: 22"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_affinity_propagation]",
            "value": 94.32622029813862,
            "unit": "iter/sec",
            "range": "stddev: 0.0005303211499610309",
            "extra": "mean: 10.601506101265178 msec\nrounds: 79"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_kmeans_digits]",
            "value": 9.847130846360653,
            "unit": "iter/sec",
            "range": "stddev: 0.004624066852924094",
            "extra": "mean: 101.55242329998941 msec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_kmeans_silhouette_analysis]",
            "value": 55.42156351880025,
            "unit": "iter/sec",
            "range": "stddev: 0.0005178964566943981",
            "extra": "mean: 18.043518380003434 msec\nrounds: 50"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_column_transformer_mixed_types]",
            "value": 23.522456202072146,
            "unit": "iter/sec",
            "range": "stddev: 0.0005000503425647791",
            "extra": "mean: 42.51256720001493 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_feature_union]",
            "value": 0.32105351987590747,
            "unit": "iter/sec",
            "range": "stddev: 0.034482594591569",
            "extra": "mean: 3.1147454804000176 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_faces_decomposition]",
            "value": 0.23124650693826462,
            "unit": "iter/sec",
            "range": "stddev: 1.0612215456368863",
            "extra": "mean: 4.324389644799988 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_gradient_boosting_categorical]",
            "value": 0.5814410187305039,
            "unit": "iter/sec",
            "range": "stddev: 0.015373734858701334",
            "extra": "mean: 1.7198649008000189 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_stack_predictors]",
            "value": 21.098356514615404,
            "unit": "iter/sec",
            "range": "stddev: 0.0009202828398073209",
            "extra": "mean: 47.397056699998075 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_voting_regressor]",
            "value": 42.86569593229492,
            "unit": "iter/sec",
            "range": "stddev: 0.0009094980010605619",
            "extra": "mean: 23.32867758823909 msec\nrounds: 34"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_rfe_digits]",
            "value": 5292.389238961645,
            "unit": "iter/sec",
            "range": "stddev: 0.000017908641929345406",
            "extra": "mean: 188.9505769224559 usec\nrounds: 286"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_select_from_model_diabetes]",
            "value": 0.9726239796326888,
            "unit": "iter/sec",
            "range": "stddev: 0.006613672748637377",
            "extra": "mean: 1.028146561199992 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_lasso_and_elasticnet]",
            "value": 22.713849100568105,
            "unit": "iter/sec",
            "range": "stddev: 0.0017385588804324707",
            "extra": "mean: 44.02600349999633 msec\nrounds: 22"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_logistic_multinomial]",
            "value": 105.3526078036001,
            "unit": "iter/sec",
            "range": "stddev: 0.0003866461755965258",
            "extra": "mean: 9.491934000003255 msec\nrounds: 48"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_quantile_regression]",
            "value": 19.855603952390187,
            "unit": "iter/sec",
            "range": "stddev: 0.006339233487926471",
            "extra": "mean: 50.36361534999401 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_confusion_matrix]",
            "value": 34.3122697308485,
            "unit": "iter/sec",
            "range": "stddev: 0.0006480137819402065",
            "extra": "mean: 29.144093580639712 msec\nrounds: 31"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_cv_indices]",
            "value": 1.4030332679065955,
            "unit": "iter/sec",
            "range": "stddev: 0.010309179622233954",
            "extra": "mean: 712.7414744000021 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_roc]",
            "value": 3.972299629321851,
            "unit": "iter/sec",
            "range": "stddev: 0.00641615286262337",
            "extra": "mean: 251.74334599998926 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_nca_classification]",
            "value": 3.759531057871132,
            "unit": "iter/sec",
            "range": "stddev: 0.025411874857110633",
            "extra": "mean: 265.99062079999385 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_mlp_alpha]",
            "value": 2.773641103177274,
            "unit": "iter/sec",
            "range": "stddev: 0.005701408568773475",
            "extra": "mean: 360.5369126000028 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_all_scaling]",
            "value": 4.287651467473151,
            "unit": "iter/sec",
            "range": "stddev: 0.019406612218419113",
            "extra": "mean: 233.22791219999317 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_discretization_strategies]",
            "value": 22.038016192897867,
            "unit": "iter/sec",
            "range": "stddev: 0.0010287845213322415",
            "extra": "mean: 45.37613509523908 msec\nrounds: 21"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_target_encoder]",
            "value": 0.5442015070385159,
            "unit": "iter/sec",
            "range": "stddev: 0.03924392674524876",
            "extra": "mean: 1.8375546319999898 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_svm_kernels]",
            "value": 67.68882060327797,
            "unit": "iter/sec",
            "range": "stddev: 0.0005913078642820925",
            "extra": "mean: 14.773488311474182 msec\nrounds: 61"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_svm_regression]",
            "value": 94.2403151806532,
            "unit": "iter/sec",
            "range": "stddev: 0.0004798819355457121",
            "extra": "mean: 10.611169944445306 msec\nrounds: 72"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_document_classification_20newsgroups]",
            "value": 0.26163700822695063,
            "unit": "iter/sec",
            "range": "stddev: 1.183997871474901",
            "extra": "mean: 3.822089263200007 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_tree_regression]",
            "value": 210.1813019635425,
            "unit": "iter/sec",
            "range": "stddev: 0.0003348485925906242",
            "extra": "mean: 4.757797152543366 msec\nrounds: 118"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_groups",
            "value": 10.945626000675093,
            "unit": "iter/sec",
            "range": "stddev: 0.001424616535625709",
            "extra": "mean: 91.36069512500455 msec\nrounds: 8"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_all_scripts",
            "value": 10.844175113690996,
            "unit": "iter/sec",
            "range": "stddev: 0.0012292607668033604",
            "extra": "mean: 92.21540500000587 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_lasso_and_elasticnet-linear_model]",
            "value": 10.637373141008979,
            "unit": "iter/sec",
            "range": "stddev: 0.0015996006047768534",
            "extra": "mean: 94.00817163636206 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_confusion_matrix-model_selection]",
            "value": 10.476299309974975,
            "unit": "iter/sec",
            "range": "stddev: 0.005233372194670318",
            "extra": "mean: 95.45355381817444 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_tree_regression-tree]",
            "value": 9.700022731827048,
            "unit": "iter/sec",
            "range": "stddev: 0.017857381621993598",
            "extra": "mean: 103.09254190908942 msec\nrounds: 11"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "xorq-labs",
            "username": "xorq-labs"
          },
          "committer": {
            "name": "xorq-labs",
            "username": "xorq-labs"
          },
          "id": "9cc9a01ba08fde69dfdd547265aa39d613fb18f2",
          "message": "Add build artifact caching and catalog sync",
          "timestamp": "2026-03-11T11:15:48Z",
          "url": "https://github.com/xorq-labs/xorq-gallery/pull/3/commits/9cc9a01ba08fde69dfdd547265aa39d613fb18f2"
        },
        "date": 1773416401989,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_list_groups",
            "value": 4532.399649709938,
            "unit": "iter/sec",
            "range": "stddev: 0.0005671817946736956",
            "extra": "mean: 220.63367692299545 usec\nrounds: 1040"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_all_scripts",
            "value": 4159.035352572387,
            "unit": "iter/sec",
            "range": "stddev: 0.0005356206827987529",
            "extra": "mean: 240.44037023669307 usec\nrounds: 2493"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_cyclical_feature_engineering]",
            "value": 2007.8760464849051,
            "unit": "iter/sec",
            "range": "stddev: 0.00014155648158246147",
            "extra": "mean: 498.038711976595 usec\nrounds: 1361"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_time_series_lagged_features]",
            "value": 2041.5848963282997,
            "unit": "iter/sec",
            "range": "stddev: 0.0009429986461922477",
            "extra": "mean: 489.8155358606227 usec\nrounds: 1952"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_tomography_l1_reconstruction]",
            "value": 2207.959951283258,
            "unit": "iter/sec",
            "range": "stddev: 0.00005306797293469387",
            "extra": "mean: 452.9067655501649 usec\nrounds: 1881"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_topics_extraction_with_nmf_lda]",
            "value": 2039.9692507023888,
            "unit": "iter/sec",
            "range": "stddev: 0.0013894909723943841",
            "extra": "mean: 490.2034673589745 usec\nrounds: 1348"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_compare_calibration]",
            "value": 2330.713062646882,
            "unit": "iter/sec",
            "range": "stddev: 0.00005028652397137891",
            "extra": "mean: 429.05324384475995 usec\nrounds: 2112"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_classification_probability]",
            "value": 2045.268718362749,
            "unit": "iter/sec",
            "range": "stddev: 0.0016784057531761675",
            "extra": "mean: 488.93330789340314 usec\nrounds: 1913"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_classifier_comparison]",
            "value": 2223.063537021876,
            "unit": "iter/sec",
            "range": "stddev: 0.00005410113299846555",
            "extra": "mean: 449.8296982279006 usec\nrounds: 1975"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_lda_qda]",
            "value": 2009.2048990183425,
            "unit": "iter/sec",
            "range": "stddev: 0.0021226368249085345",
            "extra": "mean: 497.7093179936899 usec\nrounds: 1934"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_affinity_propagation]",
            "value": 2259.386663370705,
            "unit": "iter/sec",
            "range": "stddev: 0.000048180103835735606",
            "extra": "mean: 442.59799184090645 usec\nrounds: 1961"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_kmeans_digits]",
            "value": 2246.5498643663664,
            "unit": "iter/sec",
            "range": "stddev: 0.00005295691948102846",
            "extra": "mean: 445.12699934307824 usec\nrounds: 1522"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_kmeans_silhouette_analysis]",
            "value": 2004.2847568755055,
            "unit": "iter/sec",
            "range": "stddev: 0.0023720252653093576",
            "extra": "mean: 498.9311007677909 usec\nrounds: 2084"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_column_transformer_mixed_types]",
            "value": 2296.746882126725,
            "unit": "iter/sec",
            "range": "stddev: 0.000048400929403286195",
            "extra": "mean: 435.3984358407085 usec\nrounds: 1356"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_feature_union]",
            "value": 2281.5732244876435,
            "unit": "iter/sec",
            "range": "stddev: 0.00004920852449579957",
            "extra": "mean: 438.2940636168111 usec\nrounds: 1399"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_faces_decomposition]",
            "value": 2286.869459191026,
            "unit": "iter/sec",
            "range": "stddev: 0.00006603956359003455",
            "extra": "mean: 437.2790042653975 usec\nrounds: 2110"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_gradient_boosting_categorical]",
            "value": 1948.467671513018,
            "unit": "iter/sec",
            "range": "stddev: 0.00292325776581055",
            "extra": "mean: 513.22380895521 usec\nrounds: 2010"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_stack_predictors]",
            "value": 2233.402335729175,
            "unit": "iter/sec",
            "range": "stddev: 0.00005768101449254371",
            "extra": "mean: 447.74736016093306 usec\nrounds: 1988"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_voting_regressor]",
            "value": 2235.3091211664596,
            "unit": "iter/sec",
            "range": "stddev: 0.00004992422782386353",
            "extra": "mean: 447.3654182908565 usec\nrounds: 2001"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_rfe_digits]",
            "value": 2288.6333502077114,
            "unit": "iter/sec",
            "range": "stddev: 0.00005176686737877135",
            "extra": "mean: 436.9419854470102 usec\nrounds: 1924"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_select_from_model_diabetes]",
            "value": 1675.2407961469855,
            "unit": "iter/sec",
            "range": "stddev: 0.005091901302477983",
            "extra": "mean: 596.9291115044335 usec\nrounds: 1130"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_lasso_and_elasticnet]",
            "value": 2237.7222881177,
            "unit": "iter/sec",
            "range": "stddev: 0.000050805282120835894",
            "extra": "mean: 446.8829779772037 usec\nrounds: 1226"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_logistic_multinomial]",
            "value": 2239.602808735054,
            "unit": "iter/sec",
            "range": "stddev: 0.000051195055093162624",
            "extra": "mean: 446.50774507860535 usec\nrounds: 2032"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_quantile_regression]",
            "value": 2240.058830504545,
            "unit": "iter/sec",
            "range": "stddev: 0.00004991104811461148",
            "extra": "mean: 446.4168469069907 usec\nrounds: 1940"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_confusion_matrix]",
            "value": 2236.6977065688243,
            "unit": "iter/sec",
            "range": "stddev: 0.000057956246347067235",
            "extra": "mean: 447.0876851454533 usec\nrounds: 1858"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_cv_indices]",
            "value": 2254.155304921733,
            "unit": "iter/sec",
            "range": "stddev: 0.000046207381201294285",
            "extra": "mean: 443.6251565349536 usec\nrounds: 1974"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_roc]",
            "value": 1777.5681839960778,
            "unit": "iter/sec",
            "range": "stddev: 0.004950143358658022",
            "extra": "mean: 562.5663246019295 usec\nrounds: 1947"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_nca_classification]",
            "value": 2343.5959151054303,
            "unit": "iter/sec",
            "range": "stddev: 0.00004954352845220896",
            "extra": "mean: 426.69471880992484 usec\nrounds: 2084"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_mlp_alpha]",
            "value": 2340.6746742156215,
            "unit": "iter/sec",
            "range": "stddev: 0.0000475604975282927",
            "extra": "mean: 427.22724820146476 usec\nrounds: 1946"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_all_scaling]",
            "value": 2208.6314701186952,
            "unit": "iter/sec",
            "range": "stddev: 0.000064510376995157",
            "extra": "mean: 452.7690624394927 usec\nrounds: 2066"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_discretization_strategies]",
            "value": 2232.584244377278,
            "unit": "iter/sec",
            "range": "stddev: 0.000053249506351692785",
            "extra": "mean: 447.9114293305981 usec\nrounds: 1882"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_target_encoder]",
            "value": 2243.453400321345,
            "unit": "iter/sec",
            "range": "stddev: 0.00004861098988902597",
            "extra": "mean: 445.7413734810643 usec\nrounds: 1893"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_svm_kernels]",
            "value": 1718.7021738238825,
            "unit": "iter/sec",
            "range": "stddev: 0.006161236668380143",
            "extra": "mean: 581.834372022195 usec\nrounds: 1973"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_svm_regression]",
            "value": 2108.522808746086,
            "unit": "iter/sec",
            "range": "stddev: 0.00009558156509856917",
            "extra": "mean: 474.26567825210697 usec\nrounds: 2014"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_document_classification_20newsgroups]",
            "value": 2315.240671543233,
            "unit": "iter/sec",
            "range": "stddev: 0.00004775067589571687",
            "extra": "mean: 431.9205395322664 usec\nrounds: 2011"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_tree_regression]",
            "value": 2316.859541411604,
            "unit": "iter/sec",
            "range": "stddev: 0.000048784426601051745",
            "extra": "mean: 431.6187417173876 usec\nrounds: 1479"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_cyclical_feature_engineering]",
            "value": 2.3723033155204845,
            "unit": "iter/sec",
            "range": "stddev: 0.006294681792734656",
            "extra": "mean: 421.53125760000023 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_time_series_lagged_features]",
            "value": 20.971911840532687,
            "unit": "iter/sec",
            "range": "stddev: 0.0011081942346158427",
            "extra": "mean: 47.68282489473787 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_tomography_l1_reconstruction]",
            "value": 0.14239785004873473,
            "unit": "iter/sec",
            "range": "stddev: 0.670231915605615",
            "extra": "mean: 7.022577936800005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_topics_extraction_with_nmf_lda]",
            "value": 0.0649113496897904,
            "unit": "iter/sec",
            "range": "stddev: 0.2678786077183531",
            "extra": "mean: 15.4056263624 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_compare_calibration]",
            "value": 4.039219786858872,
            "unit": "iter/sec",
            "range": "stddev: 0.0034562680314823998",
            "extra": "mean: 247.57256419999294 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_classification_probability]",
            "value": 16.6751870966913,
            "unit": "iter/sec",
            "range": "stddev: 0.0010785638409350584",
            "extra": "mean: 59.96934212500804 msec\nrounds: 16"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_classifier_comparison]",
            "value": 0.46004005934252545,
            "unit": "iter/sec",
            "range": "stddev: 0.011491817598562982",
            "extra": "mean: 2.1737237435999988 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_lda_qda]",
            "value": 43.03401681595684,
            "unit": "iter/sec",
            "range": "stddev: 0.0005590766225713848",
            "extra": "mean: 23.237431083337867 msec\nrounds: 24"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_affinity_propagation]",
            "value": 95.48950649038903,
            "unit": "iter/sec",
            "range": "stddev: 0.0009502683720136845",
            "extra": "mean: 10.47235488750431 msec\nrounds: 80"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_kmeans_digits]",
            "value": 9.73011843902001,
            "unit": "iter/sec",
            "range": "stddev: 0.000998356866839247",
            "extra": "mean: 102.77367190000177 msec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_kmeans_silhouette_analysis]",
            "value": 55.14307502258204,
            "unit": "iter/sec",
            "range": "stddev: 0.0006511958864450148",
            "extra": "mean: 18.134643372544655 msec\nrounds: 51"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_column_transformer_mixed_types]",
            "value": 22.795404564981386,
            "unit": "iter/sec",
            "range": "stddev: 0.0004430517581132925",
            "extra": "mean: 43.86849099998926 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_feature_union]",
            "value": 0.31177875517115433,
            "unit": "iter/sec",
            "range": "stddev: 0.04358905483947036",
            "extra": "mean: 3.207402632200001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_faces_decomposition]",
            "value": 0.24628765276544584,
            "unit": "iter/sec",
            "range": "stddev: 0.9300913564586589",
            "extra": "mean: 4.0602928679999994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_gradient_boosting_categorical]",
            "value": 0.5688889540041745,
            "unit": "iter/sec",
            "range": "stddev: 0.014030545290267642",
            "extra": "mean: 1.7578122987999905 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_stack_predictors]",
            "value": 21.063622665946394,
            "unit": "iter/sec",
            "range": "stddev: 0.0012097465229560292",
            "extra": "mean: 47.47521429999324 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_voting_regressor]",
            "value": 43.653228086076304,
            "unit": "iter/sec",
            "range": "stddev: 0.002284557626362638",
            "extra": "mean: 22.907813324324607 msec\nrounds: 37"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_rfe_digits]",
            "value": 5327.024275961955,
            "unit": "iter/sec",
            "range": "stddev: 0.0000744066309601039",
            "extra": "mean: 187.7220654901971 usec\nrounds: 397"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_select_from_model_diabetes]",
            "value": 0.9440927910798882,
            "unit": "iter/sec",
            "range": "stddev: 0.005908148082550264",
            "extra": "mean: 1.059217917400008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_lasso_and_elasticnet]",
            "value": 22.639401771671405,
            "unit": "iter/sec",
            "range": "stddev: 0.0007951829366907348",
            "extra": "mean: 44.17077845454804 msec\nrounds: 22"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_logistic_multinomial]",
            "value": 101.95984509175418,
            "unit": "iter/sec",
            "range": "stddev: 0.0004940780350725486",
            "extra": "mean: 9.807782653064006 msec\nrounds: 49"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_quantile_regression]",
            "value": 20.75056454064004,
            "unit": "iter/sec",
            "range": "stddev: 0.0009335824829585214",
            "extra": "mean: 48.19145994999303 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_confusion_matrix]",
            "value": 31.345068711823977,
            "unit": "iter/sec",
            "range": "stddev: 0.0011614684728669187",
            "extra": "mean: 31.902944899998904 msec\nrounds: 30"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_cv_indices]",
            "value": 1.3227180857852112,
            "unit": "iter/sec",
            "range": "stddev: 0.015425943739124478",
            "extra": "mean: 756.0189965999939 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_roc]",
            "value": 3.8048328752803693,
            "unit": "iter/sec",
            "range": "stddev: 0.013731649129828183",
            "extra": "mean: 262.82363320000286 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_nca_classification]",
            "value": 3.5403196561699475,
            "unit": "iter/sec",
            "range": "stddev: 0.023440885144128246",
            "extra": "mean: 282.4603699999898 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_mlp_alpha]",
            "value": 2.594585986291222,
            "unit": "iter/sec",
            "range": "stddev: 0.004248965609395045",
            "extra": "mean: 385.4179454000018 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_all_scaling]",
            "value": 4.2215175192367855,
            "unit": "iter/sec",
            "range": "stddev: 0.003366631450638571",
            "extra": "mean: 236.88164160000724 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_discretization_strategies]",
            "value": 21.433032714330217,
            "unit": "iter/sec",
            "range": "stddev: 0.001137827351916596",
            "extra": "mean: 46.65695300000152 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_target_encoder]",
            "value": 0.5277781489465629,
            "unit": "iter/sec",
            "range": "stddev: 0.017041748859566365",
            "extra": "mean: 1.8947355095999796 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_svm_kernels]",
            "value": 68.47958086187,
            "unit": "iter/sec",
            "range": "stddev: 0.000856847851093623",
            "extra": "mean: 14.602893116666374 msec\nrounds: 60"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_svm_regression]",
            "value": 95.92878419737593,
            "unit": "iter/sec",
            "range": "stddev: 0.00041032505553654",
            "extra": "mean: 10.424399812495011 msec\nrounds: 80"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_document_classification_20newsgroups]",
            "value": 0.2375186829143858,
            "unit": "iter/sec",
            "range": "stddev: 1.8872251385238603",
            "extra": "mean: 4.210195121200013 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_tree_regression]",
            "value": 196.9527641100311,
            "unit": "iter/sec",
            "range": "stddev: 0.0005320929104928304",
            "extra": "mean: 5.077359561409011 msec\nrounds: 57"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_groups",
            "value": 9.360536158032813,
            "unit": "iter/sec",
            "range": "stddev: 0.0015437963375775664",
            "extra": "mean: 106.8314873333236 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_all_scripts",
            "value": 9.457831815170465,
            "unit": "iter/sec",
            "range": "stddev: 0.0028384090696202662",
            "extra": "mean: 105.73247860000947 msec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_lasso_and_elasticnet-linear_model]",
            "value": 8.982918599407043,
            "unit": "iter/sec",
            "range": "stddev: 0.018080520375672853",
            "extra": "mean: 111.3223936000054 msec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_confusion_matrix-model_selection]",
            "value": 9.58795894105343,
            "unit": "iter/sec",
            "range": "stddev: 0.002035887793031041",
            "extra": "mean: 104.29748460000496 msec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_tree_regression-tree]",
            "value": 9.483040523688802,
            "unit": "iter/sec",
            "range": "stddev: 0.0019614072662868373",
            "extra": "mean: 105.4514106000056 msec\nrounds: 10"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "xorq-labs",
            "username": "xorq-labs"
          },
          "committer": {
            "name": "xorq-labs",
            "username": "xorq-labs"
          },
          "id": "915571fb33939ba6d4889fa48ba71cb3cfa9b777",
          "message": "Add build artifact caching and catalog sync",
          "timestamp": "2026-03-11T11:15:48Z",
          "url": "https://github.com/xorq-labs/xorq-gallery/pull/3/commits/915571fb33939ba6d4889fa48ba71cb3cfa9b777"
        },
        "date": 1773417249689,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_list_groups",
            "value": 4905.953302601473,
            "unit": "iter/sec",
            "range": "stddev: 0.00024428594550312285",
            "extra": "mean: 203.83398257576795 usec\nrounds: 1320"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_all_scripts",
            "value": 4368.484784810229,
            "unit": "iter/sec",
            "range": "stddev: 0.00023315368232880852",
            "extra": "mean: 228.91232298144334 usec\nrounds: 2737"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_cyclical_feature_engineering]",
            "value": 2241.077223069803,
            "unit": "iter/sec",
            "range": "stddev: 0.000052942936616925615",
            "extra": "mean: 446.2139857145177 usec\nrounds: 1400"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_time_series_lagged_features]",
            "value": 2122.510652399688,
            "unit": "iter/sec",
            "range": "stddev: 0.00048539448920603223",
            "extra": "mean: 471.1401560550053 usec\nrounds: 1602"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_tomography_l1_reconstruction]",
            "value": 2026.9091183382989,
            "unit": "iter/sec",
            "range": "stddev: 0.0005404576475715384",
            "extra": "mean: 493.3620313572916 usec\nrounds: 2041"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_topics_extraction_with_nmf_lda]",
            "value": 2148.5003795591592,
            "unit": "iter/sec",
            "range": "stddev: 0.00007211871727709228",
            "extra": "mean: 465.44092312666254 usec\nrounds: 1548"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_compare_calibration]",
            "value": 2279.095486731657,
            "unit": "iter/sec",
            "range": "stddev: 0.0006699743327382247",
            "extra": "mean: 438.77055868073904 usec\nrounds: 2062"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_classification_probability]",
            "value": 2268.8365535673042,
            "unit": "iter/sec",
            "range": "stddev: 0.00004869821776686472",
            "extra": "mean: 440.754534930114 usec\nrounds: 2004"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_classifier_comparison]",
            "value": 2189.7568276347138,
            "unit": "iter/sec",
            "range": "stddev: 0.000922579889844541",
            "extra": "mean: 456.67171230157066 usec\nrounds: 2016"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_lda_qda]",
            "value": 2283.737189877794,
            "unit": "iter/sec",
            "range": "stddev: 0.00004470927410577139",
            "extra": "mean: 437.87875611620234 usec\nrounds: 1308"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_affinity_propagation]",
            "value": 2284.086001057284,
            "unit": "iter/sec",
            "range": "stddev: 0.0000377657819850504",
            "extra": "mean: 437.8118860398025 usec\nrounds: 2106"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_kmeans_digits]",
            "value": 2279.017652759296,
            "unit": "iter/sec",
            "range": "stddev: 0.000036591263930011414",
            "extra": "mean: 438.7855437579699 usec\nrounds: 1554"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_kmeans_silhouette_analysis]",
            "value": 2138.33569966467,
            "unit": "iter/sec",
            "range": "stddev: 0.0011398971081352824",
            "extra": "mean: 467.6534185707222 usec\nrounds: 2057"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_column_transformer_mixed_types]",
            "value": 2316.2947911400706,
            "unit": "iter/sec",
            "range": "stddev: 0.000036778045347789244",
            "extra": "mean: 431.72397737327907 usec\nrounds: 2033"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_feature_union]",
            "value": 2256.6466369768327,
            "unit": "iter/sec",
            "range": "stddev: 0.00005949873430636982",
            "extra": "mean: 443.1353955086528 usec\nrounds: 1603"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_faces_decomposition]",
            "value": 2144.627146558281,
            "unit": "iter/sec",
            "range": "stddev: 0.0017526966327176775",
            "extra": "mean: 466.28151732799336 usec\nrounds: 1991"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_gradient_boosting_categorical]",
            "value": 2249.2283017378663,
            "unit": "iter/sec",
            "range": "stddev: 0.000042579377532018526",
            "extra": "mean: 444.5969309684348 usec\nrounds: 2086"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_stack_predictors]",
            "value": 2248.9508061984566,
            "unit": "iter/sec",
            "range": "stddev: 0.00004934867948669874",
            "extra": "mean: 444.6517892894078 usec\nrounds: 1998"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_voting_regressor]",
            "value": 2267.07236290545,
            "unit": "iter/sec",
            "range": "stddev: 0.000038404755681580605",
            "extra": "mean: 441.09752135058153 usec\nrounds: 2014"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_rfe_digits]",
            "value": 2032.491135527727,
            "unit": "iter/sec",
            "range": "stddev: 0.002533406460927954",
            "extra": "mean: 492.00706587109147 usec\nrounds: 2095"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_select_from_model_diabetes]",
            "value": 2300.848753499685,
            "unit": "iter/sec",
            "range": "stddev: 0.000036993234406836596",
            "extra": "mean: 434.62222298573914 usec\nrounds: 1601"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_lasso_and_elasticnet]",
            "value": 2291.13396638293,
            "unit": "iter/sec",
            "range": "stddev: 0.00003950940636637995",
            "extra": "mean: 436.46509312535954 usec\nrounds: 2051"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_logistic_multinomial]",
            "value": 2263.0603912601982,
            "unit": "iter/sec",
            "range": "stddev: 0.00003947871456895811",
            "extra": "mean: 441.8795025806378 usec\nrounds: 1550"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_quantile_regression]",
            "value": 2254.636576755338,
            "unit": "iter/sec",
            "range": "stddev: 0.000046477651167578425",
            "extra": "mean: 443.53046087769343 usec\nrounds: 2096"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_confusion_matrix]",
            "value": 1901.2065265039826,
            "unit": "iter/sec",
            "range": "stddev: 0.003831358987270397",
            "extra": "mean: 525.981783703868 usec\nrounds: 2025"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_cv_indices]",
            "value": 2250.8448976637105,
            "unit": "iter/sec",
            "range": "stddev: 0.00004406555113308477",
            "extra": "mean: 444.2776137253887 usec\nrounds: 2040"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_roc]",
            "value": 2271.6836472034934,
            "unit": "iter/sec",
            "range": "stddev: 0.000038282439190384395",
            "extra": "mean: 440.202138722541 usec\nrounds: 2004"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_nca_classification]",
            "value": 2363.2232820561626,
            "unit": "iter/sec",
            "range": "stddev: 0.000037469266709242",
            "extra": "mean: 423.1508751597661 usec\nrounds: 1562"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_mlp_alpha]",
            "value": 2361.8912409551904,
            "unit": "iter/sec",
            "range": "stddev: 0.00003754223692340834",
            "extra": "mean: 423.38952050797326 usec\nrounds: 2048"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_all_scaling]",
            "value": 2271.656624807693,
            "unit": "iter/sec",
            "range": "stddev: 0.00003932482960290947",
            "extra": "mean: 440.2073751285606 usec\nrounds: 1946"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_discretization_strategies]",
            "value": 2239.4671845828193,
            "unit": "iter/sec",
            "range": "stddev: 0.0000441455380846429",
            "extra": "mean: 446.5347859903049 usec\nrounds: 2070"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_target_encoder]",
            "value": 1828.4541769759892,
            "unit": "iter/sec",
            "range": "stddev: 0.00451963485786658",
            "extra": "mean: 546.9100689489862 usec\nrounds: 2074"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_svm_kernels]",
            "value": 2312.157728186853,
            "unit": "iter/sec",
            "range": "stddev: 0.0000374754548774744",
            "extra": "mean: 432.49644598605283 usec\nrounds: 2018"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_svm_regression]",
            "value": 2303.3246658708395,
            "unit": "iter/sec",
            "range": "stddev: 0.00003649039594509959",
            "extra": "mean: 434.1550345973136 usec\nrounds: 2110"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_document_classification_20newsgroups]",
            "value": 2347.264444472613,
            "unit": "iter/sec",
            "range": "stddev: 0.00003726390257385394",
            "extra": "mean: 426.0278394941059 usec\nrounds: 2056"
          },
          {
            "name": "tests/test_benchmarks.py::test_list_exprs[plot_tree_regression]",
            "value": 2274.0284804893645,
            "unit": "iter/sec",
            "range": "stddev: 0.00006365431893911494",
            "extra": "mean: 439.74823032330835 usec\nrounds: 2071"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_cyclical_feature_engineering]",
            "value": 2.5337262848889095,
            "unit": "iter/sec",
            "range": "stddev: 0.004193107562912332",
            "extra": "mean: 394.6756229999977 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_time_series_lagged_features]",
            "value": 21.938410466246204,
            "unit": "iter/sec",
            "range": "stddev: 0.0008852981937249496",
            "extra": "mean: 45.582153800001635 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_tomography_l1_reconstruction]",
            "value": 0.1525209396250435,
            "unit": "iter/sec",
            "range": "stddev: 0.523729940519325",
            "extra": "mean: 6.556476785799992 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_topics_extraction_with_nmf_lda]",
            "value": 0.06956783786826565,
            "unit": "iter/sec",
            "range": "stddev: 0.041957504628880024",
            "extra": "mean: 14.374458523399994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_compare_calibration]",
            "value": 4.343892952521648,
            "unit": "iter/sec",
            "range": "stddev: 0.00669476034553635",
            "extra": "mean: 230.208251199997 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_classification_probability]",
            "value": 18.951580858750656,
            "unit": "iter/sec",
            "range": "stddev: 0.0011122717774288762",
            "extra": "mean: 52.76604666666963 msec\nrounds: 18"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_classifier_comparison]",
            "value": 0.5086623190327714,
            "unit": "iter/sec",
            "range": "stddev: 0.013605048192735524",
            "extra": "mean: 1.9659407874000068 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_lda_qda]",
            "value": 45.12062550730604,
            "unit": "iter/sec",
            "range": "stddev: 0.00038868859141540915",
            "extra": "mean: 22.162813319998804 msec\nrounds: 25"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_affinity_propagation]",
            "value": 104.3951969961895,
            "unit": "iter/sec",
            "range": "stddev: 0.00024093154625639957",
            "extra": "mean: 9.578984750002443 msec\nrounds: 84"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_kmeans_digits]",
            "value": 10.466650005092822,
            "unit": "iter/sec",
            "range": "stddev: 0.0009255135260007284",
            "extra": "mean: 95.54155336362865 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_kmeans_silhouette_analysis]",
            "value": 58.218668204913435,
            "unit": "iter/sec",
            "range": "stddev: 0.0005323056052123113",
            "extra": "mean: 17.176621019228396 msec\nrounds: 52"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_column_transformer_mixed_types]",
            "value": 25.647701014201505,
            "unit": "iter/sec",
            "range": "stddev: 0.0006197335297576265",
            "extra": "mean: 38.989849400002186 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_feature_union]",
            "value": 0.337292853058988,
            "unit": "iter/sec",
            "range": "stddev: 0.0052029846465765",
            "extra": "mean: 2.9647826538000004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_faces_decomposition]",
            "value": 0.23554153177998918,
            "unit": "iter/sec",
            "range": "stddev: 1.0122152651448064",
            "extra": "mean: 4.245535776399993 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_gradient_boosting_categorical]",
            "value": 0.5970847460036979,
            "unit": "iter/sec",
            "range": "stddev: 0.022617503803188808",
            "extra": "mean: 1.6748041324000043 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_stack_predictors]",
            "value": 21.93271075703469,
            "unit": "iter/sec",
            "range": "stddev: 0.0008348304816662326",
            "extra": "mean: 45.59399935000101 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_voting_regressor]",
            "value": 45.047148056946895,
            "unit": "iter/sec",
            "range": "stddev: 0.0006444387320512205",
            "extra": "mean: 22.198963600000557 msec\nrounds: 35"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_rfe_digits]",
            "value": 5257.234922723434,
            "unit": "iter/sec",
            "range": "stddev: 0.00006499390243962863",
            "extra": "mean: 190.21406018545667 usec\nrounds: 432"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_select_from_model_diabetes]",
            "value": 0.9672461934163996,
            "unit": "iter/sec",
            "range": "stddev: 0.007194203367956635",
            "extra": "mean: 1.0338629469999887 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_lasso_and_elasticnet]",
            "value": 23.142603127839994,
            "unit": "iter/sec",
            "range": "stddev: 0.006285209517353488",
            "extra": "mean: 43.21035081818536 msec\nrounds: 22"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_logistic_multinomial]",
            "value": 111.32716048733235,
            "unit": "iter/sec",
            "range": "stddev: 0.0003015124986552408",
            "extra": "mean: 8.982533962265098 msec\nrounds: 53"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_quantile_regression]",
            "value": 21.24910186117509,
            "unit": "iter/sec",
            "range": "stddev: 0.0019626322994678353",
            "extra": "mean: 47.06081257143069 msec\nrounds: 21"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_confusion_matrix]",
            "value": 36.325656418806226,
            "unit": "iter/sec",
            "range": "stddev: 0.0005748725157420852",
            "extra": "mean: 27.528752363640375 msec\nrounds: 33"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_cv_indices]",
            "value": 1.487650767893546,
            "unit": "iter/sec",
            "range": "stddev: 0.0024302588094929846",
            "extra": "mean: 672.2007756000153 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_roc]",
            "value": 4.411126784295173,
            "unit": "iter/sec",
            "range": "stddev: 0.002729868551617976",
            "extra": "mean: 226.6994464000163 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_nca_classification]",
            "value": 1.3141270355487866,
            "unit": "iter/sec",
            "range": "stddev: 1.1448594154142853",
            "extra": "mean: 760.9614389999933 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_mlp_alpha]",
            "value": 2.868610251201671,
            "unit": "iter/sec",
            "range": "stddev: 0.006397934461274562",
            "extra": "mean: 348.6008598000012 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_all_scaling]",
            "value": 4.734612910954813,
            "unit": "iter/sec",
            "range": "stddev: 0.004179969716531171",
            "extra": "mean: 211.21050840000635 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_discretization_strategies]",
            "value": 23.97054850012529,
            "unit": "iter/sec",
            "range": "stddev: 0.0006968635623338259",
            "extra": "mean: 41.71786056521707 msec\nrounds: 23"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_target_encoder]",
            "value": 0.5600071783064154,
            "unit": "iter/sec",
            "range": "stddev: 0.028533098767436933",
            "extra": "mean: 1.785691395999993 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_svm_kernels]",
            "value": 72.07438306038057,
            "unit": "iter/sec",
            "range": "stddev: 0.0003636182653981545",
            "extra": "mean: 13.874555112906709 msec\nrounds: 62"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_svm_regression]",
            "value": 100.80504346025863,
            "unit": "iter/sec",
            "range": "stddev: 0.0002589467789714377",
            "extra": "mean: 9.920138573168115 msec\nrounds: 82"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_document_classification_20newsgroups]",
            "value": 0.30213071992646995,
            "unit": "iter/sec",
            "range": "stddev: 0.2311976516668723",
            "extra": "mean: 3.3098256286000036 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_get_exprs_for_script[plot_tree_regression]",
            "value": 211.17345978180495,
            "unit": "iter/sec",
            "range": "stddev: 0.000970471753847443",
            "extra": "mean: 4.735443559210757 msec\nrounds: 152"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_groups",
            "value": 9.952996399441219,
            "unit": "iter/sec",
            "range": "stddev: 0.012890167849211667",
            "extra": "mean: 100.47225577778187 msec\nrounds: 9"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_all_scripts",
            "value": 10.530707850439244,
            "unit": "iter/sec",
            "range": "stddev: 0.001383709011782678",
            "extra": "mean: 94.96037818182272 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_lasso_and_elasticnet-linear_model]",
            "value": 10.366927278559293,
            "unit": "iter/sec",
            "range": "stddev: 0.0006242279648827203",
            "extra": "mean: 96.46059754544467 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_confusion_matrix-model_selection]",
            "value": 10.431269208796714,
            "unit": "iter/sec",
            "range": "stddev: 0.0010193374832716472",
            "extra": "mean: 95.865611363639 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_subprocess_list_exprs[plot_tree_regression-tree]",
            "value": 10.30791100177446,
            "unit": "iter/sec",
            "range": "stddev: 0.0005435676177378033",
            "extra": "mean: 97.01286709090274 msec\nrounds: 11"
          }
        ]
      }
    ]
  }
}