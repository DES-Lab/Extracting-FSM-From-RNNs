import sys
from Comparison_with_White_Box_and_PAC import falsify_pac_based_model, falsify_refinement_based_model, run_comparison, \
    comparison_of_learning_process

# exp_name can be {tomita_[1,7]}, eg. tomita_3, tomita_5, or {bp_[1-4], eg. bp_1, bp_2}
# compare_learning_processes exp_name (table 1)
# falsify_refinement exp_name (table 2)
# falsify_pac exp_name (table 3)
# compare_pac exp_name (table 4)

help_msg = "All commands are in the form <method> <experiment_name>\n" \
           "<experiment_name> can be 'tomita_[1,7], eg. tomita_3, tomita_5, or bp_1 for balanced parentheses \n" \
           "(bp_1 and tomita_3 experiments have pretrained model which will be loaded)\n" \
           "Available experiments are:\n" \
           "\t compare_all (Table 1)\n" \
           "\t falsify_refinement (Table 2)\n" \
           "\t falsify_pac (Table 3)\n" \
            "\t compare_pac (Table 4)\n"

command_function_map = {'falsify_pac': falsify_pac_based_model, 'falsify_refinement': falsify_refinement_based_model,
                        'compare_all': run_comparison, 'compare_pac': comparison_of_learning_process}

if __name__ == '__main__':

    args = sys.argv

    if len(args) == 1 or args[1] == '-h' or args[1] == '--help' or args[1] not in command_function_map.keys():
        print(help_msg)
        exit()

    command = args[1]
    exp_name = args[2]

    command_function_map[command](exp_name)
