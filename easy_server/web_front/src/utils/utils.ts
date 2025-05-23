export function setVisableDict(
    active_key: string, visable_dict: any, single_mode: boolean = false, set_value: boolean = true
) {
    if (single_mode) {
      Object.keys(visable_dict).forEach(key => {
        if (key !== active_key) {
          visable_dict[key as keyof typeof visable_dict].value = false
        }
      })
    }
    visable_dict[active_key as keyof typeof visable_dict].value = set_value
  }