export interface TreeNode {
  label: string;
  children?: TreeNode[];
  style: string;
  [key: string]: any;
  data: { [key: string]: any };
  is_expanded?: boolean;
}
