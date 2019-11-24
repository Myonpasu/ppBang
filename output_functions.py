def readable_mod(mod_int):
    mod_names = ['NF', 'EZ', 'TD', 'HD', 'HR', 'SD', 'DT', 'RX', 'HT', 'NC', 'FL', 'AP', 'SO', 'Autopilot', 'PF',
                 '4K', '5K', '6K', '7K', '8K', 'FI']
    binary = bin(mod_int)[:1:-1]
    enabled_mods = [mod_names[digit_index] for digit_index in range(len(binary)) if int(binary[digit_index])]
    if enabled_mods:
        return ','.join(enabled_mods)
    else:
        return 'NM'
