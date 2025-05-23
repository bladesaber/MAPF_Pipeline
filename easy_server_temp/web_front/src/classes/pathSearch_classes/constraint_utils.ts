import type { BlockRoot } from "./base_root.ts";

export class BinaryConstraint {
    style: string = "BaseConstraint";
    name: string = "";
    updated_bool: boolean = false;
    block_name0: string = "";
    block_name1: string = "";
    block_property0: string = "";
    block_property1: string = "";

    constructor(
        name: string, block_name0: string, block_property0: string, block_name1: string, block_property1: string
    ) {
      this.name = name;
      this.block_name0 = block_name0;
      this.block_property0 = block_property0;
      this.block_name1 = block_name1;
      this.block_property1 = block_property1;
    }

    reset_updated() {
      this.updated_bool = false;
    }

    set_updated() {
      this.updated_bool = true;
    }

    is_updated() {
      return this.is_updated;
    }
    
    get_block_names(): string[] {
      return [this.block_name0, this.block_name1];
    }

    get_block_properties(): string[] {
      return [this.block_property0, this.block_property1];
    }

    parse_constraint(root: BlockRoot, main_name: string): [boolean, string] {
      throw new Error("Not implemented");
    }
}

export class DistanceConstraintCell extends BinaryConstraint {
    style: string = "DistanceConstraint";
    distance: number = 0;

    constructor(
        name: string, 
        block_name0: string, block_property0: string, 
        block_name1: string, block_property1: string, 
        distance: number
    ) {
      super(name, block_name0, block_property0, block_name1, block_property1);
      this.distance = distance;
    }

    parse_constraint(root: BlockRoot, main_name: string): [boolean, string] {
        // let main_block = root.get_block(main_name);
        // if (!main_block) {
        //     return [false, ""];
        // }

        // this.get_block_names().forEach((block_name) => {
        //     if (block_name != main_name) {
        //         const passive_block = root.get_block(block_name);
        //         if (!passive_block) {
        //             return [false, ""];
        //         }
        //         passive_block.set_property(this.block_property0, main_block.get_property(this.block_property0));
        //         return [true, block_name];
        //     }
        // });
        return [false, ""];
    }
}